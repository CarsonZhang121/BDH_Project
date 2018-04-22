package BDH

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}



object main {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val sf = new SparkConf().setAppName("Statistics").setMaster("local")
    val sc = new SparkContext(sf)
    val sqc = SparkSession.builder.appName("Spark CSV reader")
      .master("local")
      .getOrCreate()

    import sqc.implicits._

    //import the diagnose and summarize the ICD9_CODE by their top levels

    val diag_ICD = sqc.read.option("header","true").csv("./data/DIAGNOSES_ICD.csv")

    val diag_ICD_top = diag_ICD.withColumn("ICD9_CODE_TOP",regexp_extract($"ICD9_CODE","([EV]?\\d{3})",1))

    // Summarize the top_level ICD9_CODE by diseases categories

    val diag_ICD_new = diag_ICD_top.withColumn("ICD9_CODE_new", when($"ICD9_CODE_TOP"=== "", 1)
        .when($"ICD9_CODE_TOP">=0 and $"ICD9_CODE_TOP" <=139, 1)
        .when($"ICD9_CODE_TOP">139 and $"ICD9_CODE_TOP" <=239, 2)
        .when($"ICD9_CODE_TOP">239 and $"ICD9_CODE_TOP" <=279, 3)
        .when($"ICD9_CODE_TOP">279 and $"ICD9_CODE_TOP" <=289, 4)
        .when($"ICD9_CODE_TOP">289 and $"ICD9_CODE_TOP" <=319, 5)
        .when($"ICD9_CODE_TOP">319 and $"ICD9_CODE_TOP" <=389, 6)
        .when($"ICD9_CODE_TOP">389 and $"ICD9_CODE_TOP" <=459, 7)
        .when($"ICD9_CODE_TOP">459 and $"ICD9_CODE_TOP" <=519, 8)
        .when($"ICD9_CODE_TOP">519 and $"ICD9_CODE_TOP" <=579, 9)
        .when($"ICD9_CODE_TOP">579 and $"ICD9_CODE_TOP" <=629, 10)
        .when($"ICD9_CODE_TOP">629 and $"ICD9_CODE_TOP" <=679, 11)
        .when($"ICD9_CODE_TOP">679 and $"ICD9_CODE_TOP" <=709, 12)
        .when($"ICD9_CODE_TOP">709 and $"ICD9_CODE_TOP" <=739, 13)
        .when($"ICD9_CODE_TOP">739 and $"ICD9_CODE_TOP" <=759, 14)
        .when($"ICD9_CODE_TOP">759 and $"ICD9_CODE_TOP" <=779, 15)
        .when($"ICD9_CODE_TOP">779 and $"ICD9_CODE_TOP" <=799, 16)
        .when($"ICD9_CODE_TOP">799 and $"ICD9_CODE_TOP" <=999, 17)
        .otherwise(18) )

    // Number of ICD_9 code per admission

    val admission_count = diag_ICD_top.groupBy("HADM_ID").count()

    // Frequency of top level ICD_9 code (18 categories)

    val ICD_count = diag_ICD_new.groupBy("ICD9_CODE_new").count()

    //Assign a most frequent ICD9CODE to each admission of the subject

    /*val ICD_per_admission = diag_ICD_new.groupBy("SUBJECT_ID","HADM_ID","ICD9_CODE_new").count()
      .groupBy("SUBJECT_ID","HADM_ID")
      .agg(max(struct("count","ICD9_CODE_new")) as "struct")
      .select($"SUBJECT_ID",$"HADM_ID", $"struct.ICD9_CODE_new" as "ICD9_CODE", $"struct.count" as "count") */

    // Align all the unique ICD9_CODES to each admission


    val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

    val ICD_all_per_admission = diag_ICD_new.select($"SUBJECT_ID",$"HADM_ID", $"ICD9_CODE_new").distinct()
      .groupBy("SUBJECT_ID","HADM_ID").agg(collect_list($"ICD9_CODE_new") as "ICD9_CODES")
      .withColumn("ICD9_CODES",stringify($"ICD9_CODES"))


    // Align the top_20 ICD9_codes to each admissions

    val ICD_top20 = diag_ICD_top.groupBy("ICD9_CODE_top").count().sort(desc("count")).limit(20)

    val ICD_top20_per_admission = diag_ICD_top.join(ICD_top20,Seq("ICD9_CODE_top"))
      .select($"SUBJECT_ID",$"HADM_ID", $"ICD9_CODE_top").distinct()
      .groupBy("SUBJECT_ID","HADM_ID").agg(collect_list($"ICD9_CODE_top") as "ICD9_CODES")
      .withColumn("ICD9_CODES",stringify($"ICD9_CODES"))


    //import the noteEvents filtered by discharge summary


   val noteEvents = sqc.read
      .option("header","true")
      .option("inferSchema","true")
      .option("multiLine", "true")
      .option("mode","DROPMALFORMED")
      .csv("./data/NOTEEVENTS.csv").filter($"CATEGORY".equalTo("Discharge summary") as "struct")

    //assign only one text to each admission by using latest Row_ID method

    val Note_per_admission = noteEvents.select($"HADM_ID",$"SUBJECT_ID",$"ROW_ID",$"TEXT")
      .groupBy($"HADM_ID",$"SUBJECT_ID")
      .agg(max(struct("ROW_ID","TEXT")) as "struct")
      .select($"HADM_ID",$"SUBJECT_ID",$"struct.TEXT" as "TEXT").withColumn("TEXT",regexp_replace($"TEXT","\\[\\*\\*[^\\]]*\\*\\*\\]|<[^>]*>|[\\W]+|\\d+"," "))

    // export the csv file of one-one icd9code-clinical note

    /*val Note_ICD9 = ICD_per_admission.join(Note_per_admission,Seq("HADM_ID")).select("HADM_ID","ICD9_CODE","TEXT")

    Note_ICD9.coalesce(1)
      .write.format("csv")
      .mode("overwrite")
      .option("header", "true")
      .save("./data/NOTES_ICD9.csv") */

    // export the csv file of all-one icd9code-clinical note by using top_level ICD9CODE in 18 categories


    val Note_ALL_ICD9 = ICD_all_per_admission.join(Note_per_admission,Seq("HADM_ID")).select("HADM_ID","ICD9_CODES","TEXT")

    Note_ALL_ICD9.coalesce(1)
      .write.format("csv")
      .mode("overwrite")
      .option("header", "true")
      .save("./data/NOTES_ALL_ICD9.csv")


    // export the csv file of all-one icd9code-clinical note by using top20 ICD9code


    val Note_top20_ICD9 = ICD_top20_per_admission.join(Note_per_admission,Seq("HADM_ID")).select("HADM_ID","ICD9_CODES","TEXT")

    Note_top20_ICD9.coalesce(1)
      .write.format("csv")
      .mode("overwrite")
      .option("header", "true")
      .save("./data/NOTES_TOP20_ICD9.csv")


    //export files for graph


    ICD_count.coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .mode("overwrite")
      .save("./data/C18_ICD9CODE_count.csv")

    ICD_top20.coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .mode("overwrite")
      .save("./data/TOP20_ICD9CODE_count.csv")

  }
}
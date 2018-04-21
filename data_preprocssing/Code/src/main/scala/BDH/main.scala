package BDH

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col


object main {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val sf = new SparkConf().setAppName("Statistics").setMaster("local")
    val sc = new SparkContext(sf)
    val sqc = SparkSession.builder.appName("Spark CSV reader")
      .master("local")
      .getOrCreate()

    val noteEvents = sqc.read
      .option("header","true")
      .option("inferSchema","true")
      .option("parserLib", "univocity")
      .option("multiLine", "true")
      .option("mode", "DROPMALFORMED")
      .csv("./data/NOTEEVENTS_de.csv")

    val noteEvents_ds = noteEvents.filter(col("CATEGORY").equalTo("Discharge summary"))

    println(noteEvents_ds.count())

  }
}
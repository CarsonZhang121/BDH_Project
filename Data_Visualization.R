library(ggplot2)
library(cowplot)
library(reshape2)

# Figure 1A
ICD_code=read.csv("C18_ICD9code.csv",header=T)

ICD_code$ICD9_CODE_1=paste("category", ICD_code$ICD9_CODE_new)

p <-ggplot(ICD_code)+geom_bar(aes(x=reorder(ICD9_CODE_1,ICD9_CODE_new), y=count),stat="identity")+
  labs(y="Count",size=10)+
  #ggtitle("Count of Top-level ICD9 Codes in Categories")+
  theme(plot.title = element_text(size = 10,hjust = 0.5),axis.text.x=element_text(angle=90,size=8),axis.title.x=element_blank())



# Figure 1B-Top 20 diagnose 

ICD_code_dia=read.csv("Top20_ICD9code.csv",header=T)
#ICD_code_dia$icd_code<-reorder(ICD_code_dia$icd_code,-ICD_code_dia$frequency)

p1 <-ggplot(ICD_code_dia)+geom_bar(aes(x=ICD9_CODE_top, y=count),fill="#DD8888",stat="identity")+
  labs(x="ICD_9 Codes for Diagnosis",y="Count",size=10)+
  #ggtitle("Top20 Top_level ICD9 Codes for Diagnosis")+
  theme(plot.title = element_text(size = 10,hjust = 0.5),axis.text.x=element_text(angle=90,size=8),axis.title.x=element_blank())

# boxplot for length 

notelength=read.csv("note_length.csv",header=T)

notelength=notelength[,2:3]

colnames(notelength)=c("Approach 1","Approach 2")

p2<-ggplot(stack(notelength),aes(x=ind,y=values))+geom_boxplot()+labs(y="Length of Notes",size=10)+
  theme(axis.title.x=element_blank())#+ggtitle("Boxplot of Joined Note Lengths by two approaches")


cowplot::plot_grid(p,p1,p2,labels = "AUTO", label_size = 20,ncol = 3,nrow=1)


# Figure 2

Fig2=read.csv("Figures.csv",header=T)
Fig2_1=melt(Fig2,id="Word.Length",measure=c("Accurancy","Recall","Precision","F1.Score"))
colnames(Fig2_1)[2]="Evaluation"

p3<-ggplot(Fig2_1, aes(Word.Length,value,color=Evaluation))+ylim(0.2, 1)+labs(x="Word Sequence Length",y="Value",size=10,fill="")+geom_point(size=3)+geom_line(size=1)

# Figure 3

Fig3=read.csv("Figure3.csv",header=T)
Fig3_1=melt(Fig3[,1:3],id="word.length",measure=c("CNN","LSTM"))
p4 <- ggplot(data=Fig3_1, aes(x=word.length, y=value, fill=variable)) +
  geom_bar(stat="identity",position=position_dodge())+coord_flip() +
  scale_fill_manual(values=c('grey','#00BFC4'))+labs(y="Precision",x="Word Sequence Length",size=10,fill="")

Fig3_2=melt(Fig3[,c(1,4,5)],id="word.length",measure=c("CNN.1","LSTM.1"))

p5 <- ggplot(data=Fig3_2, aes(x=word.length, y=value, fill=variable)) +
  geom_bar(stat="identity",position=position_dodge())+coord_flip() +
  scale_fill_manual(labels=c("CNN","LSTM"),values=c('grey','#7CAE00'))+labs(y="Recall",x="Word Sequence Length",size=10,fill="")

Fig3_3=melt(Fig3[,c(1,6,7)],id="word.length",measure=c("CNN.2","LSTM.2"))
p6 <- ggplot(data=Fig3_3, aes(x=word.length, y=value, fill=variable)) +
  geom_bar(stat="identity",position=position_dodge())+coord_flip() +
  scale_fill_manual(labels=c("CNN","LSTM"),values=c('grey','#C77CFF'))+labs(y="F1 Score",x="Word Sequence Length",size=10,fill="")


cowplot::plot_grid(p4,p5,p6,labels = "AUTO", label_size = 20,ncol = 3,nrow=1)


# Figure 4

Fig4=read.csv("Figure4.csv",header=T)
Fig4_1=melt(Fig4,id="category",measure=c("precision","recall","F1.score"))
p7 <- ggplot(data=Fig4_1, aes(x=category, y=value, fill=variable)) +
  geom_bar(stat="identity",width = 0.8,position=position_dodge())+
  scale_fill_manual(labels=c("Precision","Recall","F1 Score"),values=c('#00BFC4','#7CAE00','#C77CFF'))+labs(y="Value",size=10,fill="")+
theme(axis.text.x=element_text(angle=90,size=12),axis.title.x=element_blank())+
  scale_x_discrete(limits=c("category 1","category 2","category 3","category 4","category 5","category 6","category 7","category 8","category 9","category 10","category 11","category 12","category 13","category 14","category 15","category 16","category 17","category 18","Overall"))


# Figure 5

Fig5=read.csv("Figure5.csv",header=T)
Fig5_1=melt(Fig5,id="generic.code",measure=c("precision","recall","F1.score"))
p8 <- ggplot(data=Fig5_1, aes(x=generic.code, y=value, fill=variable)) +
  geom_bar(stat="identity",width = 0.8,position=position_dodge())+
  scale_fill_manual(labels=c("Precision","Recall","F1 Score"),values=c('#00BFC4','#7CAE00','#C77CFF'))+labs(y="Value",size=10,fill="")+
  theme(axis.text.x=element_text(angle=90,size=12),axis.title.x=element_blank())+
  scale_x_discrete(limits=c("250","272","276","285","401","403","410","414","427","428","518","530","584","585","599","765","998","V300","V458","V586","Overall"))
 
#if (!requireNamespace("BiocManager", quietly = TRUE)) 
#{install.packages("BiocManager")}
#BiocManager::install("hicrep")
args=commandArgs(trailingOnly=TRUE)
require("hicrep", lib.loc='/rhome/yhu/bigdata/.rlib')
require("reshape2")

#=============================

#=============================
run <- function(){
  f1=args[1] # path coo matrix 1
  f2=args[2] # path coo matrix 2
  out=args[3] # path output
  maxdist=as.numeric(args[4])
  resol=as.numeric(args[5])
  nodefile=args[6]
  h=as.numeric(args[7])
  m1name=args[8]
  m2name=args[9] 
  #=============================
  #  f1=file.path(getwd(), 'data/output/Rao2014_10kb/SR/chr22/demo_contact_true.gz')
  #  f2=file.path(getwd(), 'data/output/Rao2014_10kb/SR/chr22/demo_contact_predict.gz') # path coo matrix 2
  #  out=file.path(getwd(), 'SCORES.txt') # path output
  #  maxdist=as.numeric(2000000)
  #  resol=as.numeric(10000)
  #  nodefile=file.path(getwd(), 'data/output/Rao2014_10kb/SR/chr22/demo.bed.gz')
  #  h=as.numeric(h)
  #  m1name='m1'
  #  m2name='m2'
  #=============================

  options(scipen=999)
  c1=2
  c2=4
  c3=5
  # read in data sets
  data1=read.table(f1)
  data2=read.table(f2)
  d1=data1[,c(c1,c2,c3)]
  d2=data2[,c(c1,c2,c3)]

  # read in nodes
  nodedata=read.table(nodefile)
  nodedata=data.frame(nodedata,nodename=as.numeric(as.character(nodedata[,4])))
  rownames(nodedata)=nodedata[,'nodename']
  nodes=as.character(nodedata[,'nodename'])

  # construct matrices
  m1=array(0,dim=c(length(nodes),length(nodes)))
  rownames(m1)=colnames(m1)=nodes
  m2=array(0,dim=c(length(nodes),length(nodes)))
  rownames(m2)=colnames(m2)=nodes

  d1[,1]=as.character(d1[,1])
  d1[,2]=as.character(d1[,2])
  d2[,1]=as.character(d2[,1])
  d2[,2]=as.character(d2[,2])
  d1cast=acast(d1, V2~V4, value.var="V5",fill=0,fun.aggregate=sum)
  d2cast=acast(d2, V2~V4, value.var="V5",fill=0,fun.aggregate=sum)
  m1[rownames(d1cast),colnames(d1cast)]=m1[rownames(d1cast),colnames(d1cast)]+d1cast
  m1[colnames(d1cast),rownames(d1cast)]=m1[colnames(d1cast),rownames(d1cast)]+t(d1cast)

  m2[rownames(d2cast),colnames(d2cast)]=m2[rownames(d2cast),colnames(d2cast)]+d2cast
  m2[colnames(d2cast),rownames(d2cast)]=m2[colnames(d2cast),rownames(d2cast)]+t(d2cast)

  m1_big=data.frame(chr='chromo',n1=as.numeric(as.character(nodedata[,2])),n2=as.numeric(as.character(nodedata[,3])),m1)
  m2_big=data.frame(chr='chromo',n1=as.numeric(as.character(nodedata[,2])),n2=as.numeric(as.character(nodedata[,3])),m2)

  colnames(m1_big)=gsub('X','',colnames(m1_big))
  colnames(m2_big)=gsub('X','',colnames(m2_big))

  m1_big[which(is.na(m1_big))]=0
  m2_big[which(is.na(m2_big))]=0

  #new code
  #========
  processed <- prep(m1_big, m2_big, resol, h, maxdist)
  SCC.out = get.scc(processed, resol, maxdist)

  # write score
  scores=data.frame(M1=m1name,M2=m2name,score=round(SCC.out[['scc']],5),sd=SCC.out[['std']], corr=SCC.out[['corr']], weight=SCC.out[['wei']])
  print(h, sep=',')
  print(scores)
  write.table(scores,file=out,quote=FALSE,row.names=FALSE,col.names=FALSE,sep='\t')
}

run()

library("tsne")

projectData <- function(max.coreferences, ncomponents) {
  data <- read.csv(paste("data-js-", ncomponents, "-", max.coreferences, ".csv", sep=""), header = FALSE) # str(data)
  sne <-tsne(data, perplexity=16)
  sne.df <- as.data.frame(sne)
  list(sne=sne, sne.df=sne.df)
}

showResults <- function(sne, sne.df, max.coreferences, ncomponents, xmin=-220, xmax=220, ymin=-220, ymax=220) { # max.coreferences <- 10; ncomponents <- 60
  labels <- read.csv("labels.csv", header = FALSE)
  names(labels) = c("class")
  info <- read.csv("info.csv", header = FALSE)
  tags <- as.character(info[,4])
  n.clusters <- nrow(unique(labels))
  ll <- labels$class # str(ll)
  ll.ac <- as.character(ll) # str(ll.ac)
  colors <- rainbow(length(unique(ll.ac)))
  names(colors) = unique(ll.ac)
  plot(sne ,t='n',xlab=NA,ylab=NA, xaxt='n', yaxt='n', xlim=c(xmin,xmax), ylim=c(ymin,ymax), 
    main=paste("T-SNE Projection of 2nd-Order Entity Features - max.coreferences = ", max.coreferences, "; ncomponents = ", ncomponents, sep=""), cex.main=0.8)
  text(sne.df, labels=ll, col=colors[ll.ac],cex=.7)
  abline(h=ymax, col="red", lty="dashed")
  abline(h=ymin, col="red", lty="dashed")
  abline(v=xmax, col="red", lty="dashed")
  abline(v=xmin, col="red", lty="dashed")
}

max.coreferences <- 16; ncomponents <- 60; 
results <- projectData(max.coreferences, ncomponents)
xmin=-220; xmax=200; ymin=-180; ymax=180
showResults(results$sne, results$sne.df, max.coreferences, ncomponents, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
plot(results$sne ,t='n',xlab=NA,ylab=NA, xaxt='n', yaxt='n', xlim=c(xmin,xmax), ylim=c(ymin,ymax), 
     main=paste("T-SNE Projection of 2nd-Order Entity Features - max.coreferences = ", max.coreferences, "; ncomponents = ", ncomponents, sep=""), cex.main=0.8)
ShowLabels(results$sne.df, 0)
ShowLabels <- function(sne.df, label) {
  labels <- read.csv("labels.csv", header = FALSE)
  names(labels) = c("class")
  info <- read.csv("info.csv", header = FALSE)
  tags <- as.character(info[,4])
  n.clusters <- nrow(unique(labels))
  ll <- labels$class # str(ll)
  ll.ac <- as.character(ll) # str(ll.ac)
  colors <- rainbow(length(unique(ll.ac)))
  names(colors) = unique(ll.ac)
  n <- nrow(labels)
  indices <- (1:n)[ll == label]
  text(sne.df[indices,], labels=ll[indices], col=colors[ll.ac[indices]], cex=.7)
  #text(sne.df[indices,], labels=tags[indices], col=colors[ll.ac[indices]],pch=pchs[ll.ac[indices]], cex=.6)
}



ShowLabels <- function(labels) {
  n.clusters <- nrow(unique(labels))
  ll <- labels$class
  ll.ac <- as.character(ll) # str(ll.ac)
  colors <- rainbow(length(unique(ll.ac)))
  names(colors) = unique(ll.ac)
  pchs <- unique(ll) %% 16 + 1
  names(pchs) = unique(ll.ac)
  
  
  n <- nrow(labels)
  indices <- (1:n)[ll == label]
  text(sne.df[indices,], labels=ll[indices], col=colors[ll.ac[indices]],pch=pchs[ll.ac[indices]], cex=.7)
  #text(sne.df[indices,], labels=tags[indices], col=colors[ll.ac[indices]],pch=pchs[ll.ac[indices]], cex=.6)
}




setwd("/home/disambiguation/data/")
data <- read.csv("data-58.csv", header = FALSE) # str(data)
data <- read.csv("data-js-60-40.csv", header = FALSE) # str(data)
labels <- read.csv("labels.csv", header = FALSE)
info <- read.csv("info.csv", header = FALSE)
tags <- as.character(info[,4])
names(labels) = c("class")

n.clusters <- nrow(unique(labels))
ll <- labels$class # str(ll)
ll.ac <- as.character(ll) # str(ll.ac)
colors <- rainbow(length(unique(ll.ac)))
names(colors) = unique(ll.ac)
pchs <- unique(ll) %% 16 + 1
names(pchs) = unique(ll.ac)

ecb = function(x,y){ plot(x, col=colors[ll.ac]);}
sne<-tsne(data, epoch_callback=ecb, perplexity=16)
sne58<-tsne(data, perplexity=16)
sne58.df <- as.data.frame(sne58)
lim <-200
ShowLabels <- function(label) {
  n <- nrow(labels)
  indices <- (1:n)[ll == label]
  text(sne.df[indices,], labels=ll[indices], col=colors[ll.ac[indices]],pch=pchs[ll.ac[indices]], cex=.7)
  #text(sne.df[indices,], labels=tags[indices], col=colors[ll.ac[indices]],pch=pchs[ll.ac[indices]], cex=.6)
}
plot(sne ,t='n', xlim=c(-lim,lim), ylim=c(-lim,lim))
ShowLabels(1)
ShowLabels(13)
ShowLabels(15)
ShowLabels(0)
ShowLabels(4)
ShowLabels(1)
ShowLabels(24)
ShowLabels(31)
ShowLabels(27)
ShowLabels(21)
ShowLabels(16)
plot(sne, main="2nd-order Features", xlab="x", ylab="y", col=colors[ll.ac], xlim=c(-lim,lim), ylim=c(-lim,lim))
plot(sne ,t='n', xlim=c(-lim,lim), ylim=c(-lim,lim))
text(sne.df, labels=ll, col=colors[ll.ac],cex=.7)
plot(sne58 ,t='n', xlim=c(-lim,lim), ylim=c(-lim,lim))
text(sne58.df, labels=ll, col=colors[ll.ac],cex=.7)
path <- "/home/disambiguation/current/entity-2-nd-wiki-topic-features.pdf"
pdf(path, width=10, height=5)
opar <- par(mfrow=c(1,2))
plot(sne ,t='n',xlab=NA,ylab=NA, xaxt='n', yaxt='n', xlim=c(-190,170), ylim=c(-150,160), main="T-SNE Projection of WIKI Topic Features", cex.main=0.8)
text(sne.df, labels=ll, col=colors[ll.ac],cex=.7)
plot(sne58 ,t='n',xlab=NA,ylab=NA, xaxt='n', yaxt='n',xlim=c(-150,170), ylim=c(-180,lim), main="T-SNE Projection of 2nd-Order Entity Features", cex.main=0.8)
text(sne58.df, labels=ll, col=colors[ll.ac],cex=.7)


dev.off()
xlim <- 180
ylim <- 160
plot(sne ,t='n', xlim=c(-200,160), ylim=c(-ylim,ylim), xlab="x", ylab="y", main="T-SNE for JS Corpus with 2nd-order Entity Features")
text(sne.df, labels=ll, col=colors[ll.ac],cex=.7)


ll.ac <- as.character(ll)
plot(sne.df,col=colors[ll.ac],pch=pchs[ll],xlim=c(-lim,lim), ylim=c(-lim,lim))
plot(sne ,t='n',xlim=c(-150,160), ylim=c(-160,180))
text(sne.df, labels=ll.ac, col=colors[ll.ac], cex=.8)

setwd("/tmp/")
Y <- read.csv("Y.csv", header = FALSE)
data.2nd <- cbind(data,Y)
str(data)
str(data.2nd)
labels <- read.csv("labels.csv", header = FALSE)
names(labels) = c("class")
n.clusters <- nrow(unique(labels))

colors <- rainbow(n.clusters)
AssignColor <- function(x) {
  color <- "black"
  if (x == 8) {color <- "red"}
  color
}
AssignColor <- function(x) {
  if (x == 0) {x <- 4}
  if (x == 4) {x <- 0}
  colors[x+1]
}
AssignPCH <- function(x) {
  if (x == 0) {x <- 4}
  if (x == 4) {x <- 0}
  y <- x %% 16
  y
}
col <- sapply(labels$class, AssignColor)
col <- rainbow(length(unique(labels$class)))
names(colors) = unique(labels$class)
pch <- sapply(labels$class, AssignPCH)
################################################################################
# t-sne                                                                          #
################################################################################
library("tsne")
ecb = function(x,y){ plot(x, col=col[labels$class], pch=pch);}
sne<-tsne(data, epoch_callback = ecb, perplexity=16)
?tsne
lim <-200
plot(sne, main="2nd-order Features", xlab="x", ylab="y", col=col[labels$class], xlim=c(-lim,lim), ylim=c(-lim,lim), pch=pch)
ShowLabels <- function(label) {
  n <- nrow(labels)
  indices <- (1:n)[ll == label]
  text(sne.df[indices,], labels=ll[indices], col=colors[as.character(ll[indices])],pch=pchs[as.character(ll[indices])], cex=.7)
}
ll.ac <- as.character(ll)
plot(sne.df,col=colors[ll.ac],pch=pchs[ll],xlim=c(-lim,lim), ylim=c(-lim,lim))
plot(sne ,t='n',xlim=c(-150,160), ylim=c(-160,180))
text(sne.df, labels=ll.ac, col=colors[ll.ac], cex=.8)
ShowLabels(30)
ShowLabels(0)
ShowLabels(4)
ShowLabels(13)
ShowLabels(1)
ShowLabels(28)
ShowLabels(15)
ShowLabels(24)
ShowLabels(31)
ShowLabels(27)
ShowLabels(21)
ShowLabels(16)
colors <- rainbow(length(unique(ll)))
names(colors) = as.character(unique(ll))
pchs <- unique(ll) %% 16 + 1
names(pchs) = as.character(unique(ll))
label = 0
str(ll)
plot(sne ,t='n',xlim=c(-lim,lim), ylim=c(-lim,lim))
text(sne,labels=ll)
attributes(sne)
sne.df <- as.data.frame(sne)
str(sne.df)
plot(sne.df ,t='n',xlim=c(-lim,lim), ylim=c(-lim,lim))
text(sne.df,labels=ll)
sne.df[1,]
colors[0]

text(-100,80, labels='x')
abline(v=.5,col='red',lty='dashed')
abline(h=.5,col='blue',lty='dashed')
plot(-1:1, -1:1, type = "n", xlab = "Re", ylab = "Im")
K <- 16; text(exp(1i * 2 * pi * (1:K) / K), col = 2)
df.sne <- data.frame(sne)
df.sne[1,1]

colors = rainbow(length(unique(iris$Species)))
names(colors) = unique(iris$Species)
ecb = function(x,y){ plot(x,t='n'); text(x,labels=iris$Species, col=colors[iris$Species]) }
tsne_iris = tsne(iris[,1:4], epoch_callback = ecb, perplexity=50)
plot(tsne_iris,col=colors[iris$Species])
text(tsne_iris,labels=iris$Species, col=colors[iris$Species])


# compare to PCA
dev.new()
pca_iris = princomp(iris[,1:4])$scores[,1:2]
plot(pca_iris, t='n')
text(pca_iris, labels=iris$Species,col=colors[iris$Species])


text(sne[indices], labels=ll[indices])
ll <- labels$class
indices = ll == 0
# 1st-order features:
sne<-tsne(data, epoch_callback = ecb, perplexity=16)
# 2nd-order features:
sne.2nd<-tsne(data.2nd, epoch_callback = ecb, perplexity=16)
?text

o.par <- par(mfrow=c(1,2))
lim <-200
plot(sne, main="1st-order Features", xlab="x", ylab="y", col=col, xlim=c(-lim,lim), ylim=c(-lim,lim), pch=pch)
lim.2nd <-200
plot(sne.2nd, main="2nd-order Features", xlab="x", ylab="y", col=col, xlim=c(-lim.2nd,lim.2nd), ylim=c(-lim.2nd,lim.2nd), pch=pch)
par(o.par)
cluster.obj <- bclust(data.matrix(Y), transformed.par = c(0, -50, log(16), 0, 0, 0))
plot(cluster.obj)
ditplot(cluster.obj)
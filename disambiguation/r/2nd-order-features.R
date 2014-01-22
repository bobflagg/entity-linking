setwd("/tmp/")
data <- read.csv("data-58.csv", header = FALSE)
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
plot(sne ,t='n',xlim=c(-lim,lim), ylim=c(-lim,lim))
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
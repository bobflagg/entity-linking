################################################################################
## R-script for visualizing entity disambiguation clustering results, using   ##
## t-Distributed Stochastic Neighbor Embedding.                               ##
################################################################################
# http://www.r-tutor.com/gpu-computing/clustering/distance-matrix
install.packages("fpc")
install.packages("tsne")
install.packages("rattle")
# Set working directory:
WORKING_DIR <- "/opt/disambiguation"
setwd(WORKING_DIR)
################################################################################
# prepare the data                                                             #
################################################################################
LabelIt <- function(x) {
  if (grepl("hillary",x[2])) return(x[1])
  return("")
}
ColorIt <- function(x) {
  if (grepl("clinton",x)) {
    if (grepl("bill",x)) return("red")
    if (grepl("hillary",x)) return("green")
    if (grepl("chelsea",x)) return("blue")
    return("yellow")
  }
  return("grey")
}
PchIt <- function(x) {
  if (grepl("clinton",x)) return(16)
  return(1)
}
LoadData <- function(contraint="clinton") {
  data = read.csv("data-test/clinton.csv", header=F, sep=" ") # str(data)
  features = read.csv("data-test/features.tsv", header=F, sep="\t") # str(data)
  names(features) <- c("index","subtype","entity","value")
  labels = read.csv("data-test/labels.txt", header=F, row.names = NULL, sep="\t") # ?read.csv
  names(labels) <- c("doc", "entity")
  labels$doc <- as.character(labels$doc)
  labels$entity <- as.character(labels$entity)
  indices <- (1:nrow(data))[grepl("clinton|bush",labels$entity)]
  data <- data[indices,]
  labels <- labels[indices,]
  text <- apply(labels, 1, LabelIt)
  col <- sapply(labels$entity, ColorIt)
  pch <- sapply(labels$entity, PchIt)  
  list(data, features, labels, text, col, pch)
}
results <- LoadData(); i <- 1; data <- results[[i]]; i <- i + 1; features <- results[[i]]; i <- i + 1; labels <- results[[i]]; i <- i + 1; text <- results[[i]]; i <- i + 1; col <- results[[i]]; i <- i + 1; pch <- results[[i]]
CompareFeatures <- function(first.index, second.index, type="A") { # first.index <- 199; second.index <- 667;
  first.data <- features[features$index==first.index,]
  second.data <- features[features$index==second.index,]
  merged.data <- merge(first.data, second.data, by=c("subtype", "entity"), all=TRUE)[, c(-3,-5)]
  first.header = paste(labels[first.index+1,1], labels[first.index+1,2],sep=":")
  second.header = paste(labels[second.index+1,]$doc, labels[second.index+1,]$entity,sep=":")
  names(merged.data) <- c("subtype","entity",paste("<",first.header,">",sep=""), paste("<",second.header,">",sep=""))
  if (type == "D") {
    merged.data <- merged.data[!complete.cases(merged.data),]
  }
  if (type == "E") {
    merged.data <- merged.data[complete.cases(merged.data),]
  }
  merged.data[is.na(merged.data[,3]),3] <- "-"
  merged.data[is.na(merged.data[,4]),4] <- "-"
  merged.data
}
CompareFeatures(1, 4, type="A")

################################################################################
# t-sne                                                                          #
################################################################################
library("tsne")
#ecb = function(x,y){ plot(x, col=col); text(x,labels=text, cex = .5) }
ecb = function(x,y){ plot(x, col=col);}
# Compute t-sne model:
sne<-tsne(data, epoch_callback = ecb, perplexity=24)
path <- "/opt/disambiguation/clinton-sne-projections.pdf"
pdf(path, width=6.25, height=5)
lim <- 200
plot(sne, main="t-SNE Projection of Clinton-Related Mentions", xlab="x", ylab="y", col=col, xlim=c(-lim,lim), ylim=c(-lim,lim), pch=pch)
legend(x="bottomright",legend=c("Bill", "Hillary", "Chelsea", "Clinton"),pch=16, cex=.8,col=c("red","green","blue","yellow"))
dev.off()
text(sne[,1], sne[,2], text, cex = .5)
plot(sne, main="t-SNE Projection of Clinton-Related Mentions", xlab="x", ylab="y", col=col)

unique(labels$doc)

library("tsne")
ecb = function(x,y){ plot(x, col=col); text(x,labels=text, cex = .5) }
# Compute t-sne model:
sne<-tsne(data, epoch_callback = ecb, perplexity=50)
plot(sne, main="t-SNE Projection of Clinton-Related Mentions", xlab="x", ylab="y", col=col, xlim=c(-60,60), ylim=c(-60,60))
text(sne[,1], sne[,2], text, cex = .5)

ClusterSNE <- function(n.clusters) {
  colors <- rainbow(n.clusters)
  fit <- kmeans(sne, n.clusters)
  sne.clustered <- data.frame(sne, fit$cluster)
  path <- "sne-kmeans.pdf"
  pdf(path, width=6.25, height=5)
  plot(sne, main="Clustered t-SNE Projection", xlab="x", ylab="y", col=colors[fit$cluster], pch=fit$cluster, xlim=c(-60,60), ylim=c(-60,60))
  dev.off()
  sne.clustered.labels <- data.frame(labels, fit$cluster)
  sne.clustered.labels[order(fit$cluster),]
  names(sne.clustered.labels) <- c("name", "cluster")
  write.csv(sne.clustered.labels[order(fit$cluster),], file="sne-kmeans.csv", row.names = F)
}
ClusterSNE(40)
n.clusters <- 48
colors <- rainbow(n.clusters)
fit <- kmeans(sne, n.clusters)
sne.clustered <- data.frame(sne, fit$cluster)
plot(sne, main="Clustered t-SNE Projection", xlab="x", ylab="y", col=colors[fit$cluster], pch=(fit$cluster %% 16), xlim=c(-lim,lim), ylim=c(-lim,lim))

ShowClusters <- function(phrase) { # phrase <- "hillary"
  SelectIt <- function(x) {grepl(phrase,x)}
  indices <- sapply(labels, SelectIt)
  data.frame(id=(1:nrow(data) - 1)[indices], label=labels[indices], cluster=fit$cluster[indices])
}
ShowNames <- function(index) { # index <- 13
  indices <- fit$cluster==index
  data.frame(id=(1:nrow(data))[indices], doc=labels[indices,1], entity=labels[indices,2], cluster=fit$cluster[indices])
}

indices <- order(fit$cluster)
data.clustered <- data.frame(id=(1:nrow(data))[indices], doc=labels[indices,1], entity=labels[indices,2], cluster=fit$cluster[indices])
write.csv(data.clustered, file="t-sne-clusters.csv", row.names = F)

ShowNames(1)

plot(sne, main="t-SNE Projection of Clinton-Related Mentions", xlab="x", ylab="y", col=colors[fit$cluster], xlim=c(-60,60), ylim=c(-60,60))

clusplot(sne.clustered, fit$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
ShowClusters("hillary")
ShowNames(13)

################################################################################
# K-Means                                                                      #
################################################################################
fit <- kmeans(data, 8)
table(fit$cluster)
indices <- order(fit$cluster)
data.clustered <- data.frame(id=(1:nrow(data))[indices], doc=labels[indices,1], entity=labels[indices,2], cluster=fit$cluster[indices])
write.csv(data.clustered, file="kmeans-clusters-8.csv", row.names = F)

nrow(data)

data.clustered <- data.frame(data, fit$cluster)
clustered.labels <- data.frame(labels, fit$cluster)
names(clustered.labels) <- c("name", "cluster")
write.csv(clustered.labels[order(fit$cluster),], file="kmeans-24.csv", row.names = F)
ShowClusters("hillary")
ShowNames(13)
CompareFeatures(0, 57)
CompareFeatures(9, 30, all=F)
ShowDiffs(9, 30)
# get cluster means 
aggregate(data,by=list(fit$cluster),FUN=mean)
# append cluster assignment
data.clustered <- data.frame(data, fit$cluster)
# Cluster Plot against 1st 2 principal components
# vary parameters for most readable graph
library(cluster) 
clusplot(data.clustered, fit$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(data, fit$cluster)
ShowClusters <- function(phrase) { # phrase <- "hillary"
  SelectIt <- function(x) {grepl(phrase,x)}
  indices <- sapply(labels, SelectIt)
  data.frame(id=(1:nrow(data) - 1)[indices], label=labels[indices], cluster=fit$cluster[indices])
}
ShowNames <- function(index) {
  indices <- fit$cluster==index
  data.frame(id=(1:nrow(data) - 1)[indices], label=labels[indices], cluster=fit$cluster[indices])
}
labels[2:8]
################################################################################
# Hierarchical Clustering                                                      #
################################################################################
hclust <- hclusterpar(data, method="euclidean", link="ward", nbproc=1)
require(ggplot2, quietly=TRUE)
require(ggdendro, quietly=TRUE)
ddata <- dendro_data(hclust, type="rectangle")
g <- ggplot(segment(ddata))
g <- g + geom_segment(aes(x = y, y = x, xend = yend, yend = xend))
g <- g + scale_y_discrete(labels = text)
g <- g + labs(x="Height", y="Observation")
g <- g + ggtitle(expression(atop("Cluster Dendrogram")))
print(g)
g <- g + ggtitle(expression(atop("Cluster Dendrogram", atop(italic("Rattle 2013-Nov-21 12:48:00 rflagg")))))

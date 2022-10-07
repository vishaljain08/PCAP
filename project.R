library(readr)
library(dplyr)
library(xgboost)
library(doSNOW)
library(caret)
library(FactoMineR)
library(factoextra)
library(ranger)

setwd("D:/Downloads/CSV-03-11/03-11")


df <- read.csv('dataReduced.csv', sep=',')
df <- as.data.frame(df)

pca_data <- select(df, -'Unnamed..0', -'Flow.ID', -'Source.IP', -'Destination.IP', -'Timestamp', -'Label',
                  -'Flow.Bytes.s', -'Flow.Packets.s', -'SimillarHTTP', -'Fwd.Avg.Bytes.Bulk', -'Fwd.Avg.Packets.Bulk',
                  -'Fwd.Avg.Bulk.Rate', -'Bwd.Avg.Bytes.Bulk', -'Bwd.Avg.Packets.Bulk', -'Bwd.Avg.Bulk.Rate', -'ECE.Flag.Count',
                  -'PSH.Flag.Count', -'FIN.Flag.Count', -'Bwd.URG.Flags', -'Fwd.URG.Flags', -'Bwd.PSH.Flags')

centesc <- scale(pca_data, center=TRUE, scale=TRUE)

res.pca <- PCA(centesc, scale.unit = FALSE, graph = FALSE, ncp=15)
eig.val <- get_eigenvalue(res.pca) # Obtenemos los landas del pca
eig.val
fviz_eig(res.pca,addlabels = TRUE)

# We select the top 15 features and we get a list with the colnamnes. Dim 1-2
a <- fviz_contrib(res.pca, choice = "var", axes = 1:2, top = 25)
data_a <- a$data
list_a <- rownames(data_a[data_a$contrib >2,])

b <- fviz_contrib(res.pca, choice = "var", axes = 2:3, top = 25)
data_b <- b$data
list_b <- rownames(data_b[data_b$contrib > 2,])

merged.lists <- c(list_a, list_b)
merged.lists <- unique(merged.lists)

merged.lists <- c(merged.lists, "Label")
merged.lists

final.list <- select(df, c(merged.lists))

smallIndex <- createDataPartition(final.list$Label,
                                  p = .01,
                                  list = FALSE,
                                  times = 1)

small <- final.list[smallIndex,]

trainIndex <- createDataPartition(small$Label,
                                  p = .7,
                                  list = FALSE,
                                  times = 1)

small.train <- small[trainIndex,]
small.test <- small[-trainIndex,]


train.control <- trainControl(number = 10,
                              repeats = 1,
                              method = "repeatedcv",
                              verboseIter = TRUE)

cl <- makeCluster(8, type = "SOCK")
registerDoSNOW(cl)

forestFit <- train(Label ~ .,
                   data = small.train,
                   method = "ranger",
                   trControl = train.control,
                   verbose = TRUE)

stopCluster(cl)

(forestFit)


preds <- predict(forestFit, small.test)

score.test.augmented <-
  small.test %>%
  mutate(pred = predict(forestFit, small.test),
         obs = Label)

defaultSummary(as.data.frame(score.test.augmented))

score.test.augmented %>% 
  group_by(Label) %>%
  summarise(no_rows = length(Label))

score.test.augmented %>% 
  group_by(pred) %>%
  summarise(no_rows = length(pred))

xgb.plot.tree(model = forestFit$finalModel, trees = 1)


if (score.test.augmented$Label == score.test.augmented$pred) {
  correct
}

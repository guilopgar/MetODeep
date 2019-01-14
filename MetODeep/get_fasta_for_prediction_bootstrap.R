library(bio3d)
library(caret)

load('MetOData.RData')

dir_dest <- 'fasta_files_bs/'

ACC_ids <- unique(MetOData$ACC)

insert.at <- function(a, pos, ...){
  dots <- list(...)[[1]]
  stopifnot(length(dots)==length(pos))
  result <- vector("list",2*length(pos)+1)
  result[c(TRUE,FALSE)] <- split(a, cumsum(seq_along(a) %in% (pos+1)))
  result[c(FALSE,TRUE)] <- dots
  unlist(result)
}


# Split proteins for training and testing
# 100 boostrap data

set.seed(1)
train_ids_bs <- createResample(y = ACC_ids, times = 100)

for (j in 1:100){
  print(paste('Boostrap', j))

  train_ids <- train_ids_bs[[j]]
  
  for (i in 1:length(ACC_ids)){
    sq <- get.seq(ACC_ids[i])
    # sq_char <- paste(sq$ali, collapse = '')
    if (i %in% train_ids){
      sq_raw <- sq$ali
      pos_mets <- MetOData$Met[MetOData$ACC == ACC_ids[i] & MetOData$Oxidable == 'Yes']
      sq_raw <- insert.at(sq_raw, pos_mets, rep('#', length(pos_mets)))
      sq$ali <- sq_raw
      dest_file <- paste0('train_', ACC_ids[i])
    } else {
      dest_file <- paste0('test_', ACC_ids[i])
    }
    write.fasta(alignment = sq, file = paste0(dir_dest, paste0(dest_file, '.fasta')))
  }

  setwd(dir_dest)
  system(paste0('cat train_* > all_train_MetOx_bs_', j, '.fasta'))
  system(paste0('cat test_* > all_test_MetOx_bs_', j, '.fasta'))
  system('rm test_*')
  system('rm train_*')
  setwd('..')
}
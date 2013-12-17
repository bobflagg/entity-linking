from model import build_lda_topic_model

path = "/home/disambiguation/data-sets/js-test-context"
path = "/home/disambiguation/data-sets/mycorpus.txt"
description = "js test context corpus"
description = "Toy corpus example from  Deerwester et al. (1990)"
num_topics = 100
corpus, model = build_lda_topic_model(path, description, num_topics)
import sys
from transformers import TFBertForTokenClassification

model_dir = sys.argv[1]

tf_model = TFBertForTokenClassification.from_pretrained(model_dir, from_pt=True)
tf_model.save_pretrained(model_dir)
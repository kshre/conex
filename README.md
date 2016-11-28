# conex
Concept Extraction in Clinical Notes

vectors.txt is the word vector embeddings obtained from GloVE trained on MIMIC

training command:
python biscuit/galen train --txt "data/train/txt/*.txt" --annotations "data/train/con/*" --model models/word-ls --format i2b2


prediction command:
python biscuit/galen predict --txt "data/test/txt/*.txt" --model models/word-ls --out data/predictions/conex-test --format i2b2


evaluation command:
python biscuit/galen evaluate --predictions data/predictions/conex-test/ --gold data/test/con/ --format i2b2

Using code originally from https://github.com/wboag/conex

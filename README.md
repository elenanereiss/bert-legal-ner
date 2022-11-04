# How to fine-tune BERT on [German LER Dataset](https://github.com/elenanereiss/Legal-Entity-Recognition)

Based on the [scripts](https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification) for token classification on the GermEval 2014 (German NER) dataset. The dataset includes two different versions of annotations, 19 fine- and 7 coarse-grained labels. All labels are in BIO format. Distribution of coarse- and fine-grained classes in the dataset:

|   |         | **Coarse-grained classes** | **#**      | **%**   |    |         | **Fine-grained classes** | **#**      | **%**   |
|---|---------|----------------------------|------------|---------|----|---------|--------------------------|------------|---------|
| 1 | **PER** | _Person_                   | 3,377      | 6.30    | 1  | **PER** | _Person_                 | 1,747      | 3.26    |
|   |         |                            |            |         | 2  | **RR**  | _Judge_                  | 1,519      | 2.83    |
|   |         |                            |            |         | 3  | **AN**  | _Lawyer_                 | 111        | 0.21    |
| 2 | **LOC** | _Location_                 | 2,468      | 4.60    | 4  | **LD**  | _Country_                | 1,429      | 2.66    |
|   |         |                            |            |         | 5  | **ST**  | _City_                   | 705        | 1.31    |
|   |         |                            |            |         | 6  | **STR** | _Street_                 | 136        | 0.25    |
|   |         |                            |            |         | 7  | **LDS** | _Landscape_              | 198        | 0.37    |
| 3 | **ORG** | _Organization_             | 7,915      | 14.76   | 8  | **ORG** | _Organization_           | 1,166      | 2.17    |
|   |         |                            |            |         | 9  | **UN**  | _Company_                | 1,058      | 1.97    |
|   |         |                            |            |         | 10 | **INN** | _Institution_            | 2,196      | 04.09   |
|   |         |                            |            |         | 11 | **GRT** | _Court_                  | 3,212      | 5.99    |
|   |         |                            |            |         | 12 | **MRK** | _Brand_                  | 283        | 0.53    |
| 4 | **NRM** | _Legalnorm_                | 20,816     | 38.81   | 13 | **GS**  | _Law_                    | 18,52      | 34.53   |
|   |         |                            |            |         | 14 | **VO**  | _Ordinance_              | 797        | 1.49    |
|   |         |                            |            |         | 15 | **EUN** | _European legal norm_    | 1,499      | 2.79    |
| 5 | **REG** | _Case-by-case regulation_  | 3,47       | 6.47    | 16 | **VS**  | _Regulation_             | 607        | 1.13    |
|   |         |                            |            |         | 17 | **VT**  | _Contract_               | 2,863      | 5.34    |
| 6 | **RS**  | _Court decision_           | 12,58      | 23.46   | 18 | **RS**  | _Court decision_         | 12,58      | 23.46   |
| 7 | **LIT** | _Legal literature_         | 3,006      | 5.60    | 19 | **LIT** | _Legal literature_       | 3,006      | 5.60    |
|   |         | **Total**                  | **53,632** | **100** |    |         | **Total**                | **53,632** | **100** |

## Training with 19 fine-grained labels

How to fine-tune the BERT with 19 labels see Jupyter Notebook [Train_BERT_on_19_labels.ipynb](https://github.com/elenanereiss/bert-for-german-legal-ner/blob/main/Train_BERT_on_19_labels.ipynb)

## Training with 7 coarse-grained labels

How to fine-tune the BERT with 7 labels see Jupyter Notebook [Train_BERT_on_7_labels.ipynb](https://github.com/elenanereiss/bert-for-german-legal-ner/blob/main/Train_BERT_on_7_labels.ipynb)

## Run in terminal

You can execute a bash in terminal `./fine_run.sh BERT_MODEL` for training on German LER Dataset with fine-grained labels, e.g. [bert-base-german-cased](https://huggingface.co/bert-base-german-cased):

```bash
./fine_run.sh bert-base-german-cased
``` 

Or you can execute `./coarse_run.sh BERT_MODEL` for training with coarse-grained labels:

```bash
./coarse_run.sh bert-base-german-cased
``` 

Note that is a Pytorch version. With Tensorflow you have to change line 39 to `python3 run_tf_ner.py ...`. Make sure to run `chmod a+x fine_run.sh` to make your script executable.
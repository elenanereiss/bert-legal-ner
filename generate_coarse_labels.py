import sys
import codecs

filename = sys.argv[1]
filename_coarse = sys.argv[2]

def generate_coarse_labels(filename, filename_coarse):
    sentence_counter = 0
    fc = open(filename_coarse, 'w', encoding='utf-8')

    for line in codecs.open(filename, 'r', 'utf8'):
        line = line.strip()
        if line:
            word, label = line.split(' ')
            # generate coarse-grained tags
            new_label = ""
            if label == 'O': new_label = label
            else:
                bio, fine_tag = label.split("-")
                if fine_tag  in ['PER', 'RR', 'AN']: new_label = bio + '-PER'
                elif fine_tag  in ['LD', 'ST', 'STR', 'LDS']: new_label = bio + '-LOC'
                elif fine_tag  in ['ORG', 'UN', 'INN', 'GRT', 'MRK']: new_label = bio + '-ORG'
                elif fine_tag  in ['GS', 'VO', 'EUN']: new_label = bio + '-NRM'
                elif fine_tag  in ['VS', 'VT']: new_label = bio + '-REG'
                else: new_label = label
            fc.write(word + ' ' + new_label + '\n')
    
        else:
            fc.write('\n')
            sentence_counter += 1

    fc.close()
    print('Coarse-grained labels created in {}'.format(filename_coarse))

generate_coarse_labels(filename, filename_coarse)
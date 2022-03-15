WIKI_BPE_FOLDER = '/gscratch/zlab/margsli/data/wikipedia/bpe/'
WIKI_IN_FILENAME = 'wikipedia-en-0.bpe_raw'
WIKI_TRAIN_FILENAME = 'wikipedia-en-0.bpe'
WIKI_VAL_FILENAME = 'valid-wikipedia-en.bpe'
WIKI_TEST_FILENAME = 'test-wikipedia-en.bpe'
WIKI_METADATA_FILENAME = ''
VALID_TOKENS = 10**7
TEST_TOKENS = 10**7

cur_lines = []
lines = {}
count = 0
with open(WIKI_BPE_FOLDER + WIKI_IN_FILENAME, 'r') as f:
    for l in f:
        tokens = l.strip().split()
        cur_lines.append(l)
        count += len(tokens)
        if 'valid' not in lines and count >= VALID_TOKENS:
            count = count - VALID_TOKENS
            lines_to_add = cur_lines
            cur_lines = []
            if count > 0:
                last_line, first_cur_line = ' '.join(tokens[:-count]), ' '.join(tokens[-count:])
                lines_to_add[-1] = last_line
                cur_lines.append(first_cur_line)
            lines['valid'] = lines_to_add
        if 'test' not in lines and count >= TEST_TOKENS:
            count = count - TEST_TOKENS
            lines_to_add = cur_lines
            cur_lines = []
            if count > 0:
                last_line, first_cur_line = ' '.join(tokens[:-count]), ' '.join(tokens[-count:])
                lines_to_add[-1] = last_line
                cur_lines.append(first_cur_line)
            lines['test'] = lines_to_add
    lines['train'] = cur_lines

with open(WIKI_BPE_FOLDER + WIKI_TRAIN_FILENAME, 'w') as wf:
    for l in lines['train']:
        wf.write(l)

with open(WIKI_BPE_FOLDER + WIKI_VAL_FILENAME, 'w') as wf:
    for l in lines['valid']:
        wf.write(l)
        
with open(WIKI_BPE_FOLDER + WIKI_TEST_FILENAME, 'w') as wf:
    for l in lines['test']:
        wf.write(l)
        
        
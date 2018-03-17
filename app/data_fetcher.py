from conllu import parse


class DataFetcher:
    def __init__(self):
        pass

    @staticmethod
    def data_sample():
        """
        A sample method that returns two sentences of .conllu format tagging
        :return: a list of strings with the .conllu format for each sentence
        """
        data = """
            # sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0002
            # text = [This killing of a respected cleric will be causing us trouble for years to come.]
            1	[	[	PUNCT	-LRB-	_	10	punct	10:punct	SpaceAfter=No
            2	This	this	DET	DT	Number=Sing|PronType=Dem	3	det	3:det	_
            3	killing	killing	NOUN	NN	Number=Sing	10	nsubj	10:nsubj	_
            4	of	of	ADP	IN	_	7	case	7:case	_
            5	a	a	DET	DT	Definite=Ind|PronType=Art	7	det	7:det	_
            6	respected	respected	ADJ	JJ	Degree=Pos	7	amod	7:amod	_
            7	cleric	cleric	NOUN	NN	Number=Sing	3	nmod	3:nmod	_
            8	will	will	AUX	MD	VerbForm=Fin	10	aux	10:aux	_
            9	be	be	AUX	VB	VerbForm=Inf	10	aux	10:aux	_
            10	causing	cause	VERB	VBG	VerbForm=Ger	0	root	0:root	_
            11	us	we	PRON	PRP	Case=Acc|Number=Plur|Person=1|PronType=Prs	10	iobj	10:iobj	_
            12	trouble	trouble	NOUN	NN	Number=Sing	10	obj	10:obj	_
            13	for	for	ADP	IN	_	14	case	14:case	_
            14	years	year	NOUN	NNS	Number=Plur	10	obl	10:obl	_
            15	to	to	PART	TO	_	16	mark	16:mark	_
            16	come	come	VERB	VB	VerbForm=Inf	14	acl	14:acl	SpaceAfter=No
            17	.	.	PUNCT	.	_	10	punct	10:punct	SpaceAfter=No
            18	]	]	PUNCT	-RRB-	_	10	punct	10:punct	_

            # sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0002
            # text = [This killing of a respected cleric will be causing us trouble for years to come.]
            1	[	[	PUNCT	-LRB-	_	10	punct	10:punct	SpaceAfter=No
            2	This	this	DET	DT	Number=Sing|PronType=Dem	3	det	3:det	_
            3	killing	killing	NOUN	NN	Number=Sing	10	nsubj	10:nsubj	_
            4	of	of	ADP	IN	_	7	case	7:case	_
            5	a	a	DET	DT	Definite=Ind|PronType=Art	7	det	7:det	_
            6	respected	respected	ADJ	JJ	Degree=Pos	7	amod	7:amod	_
            7	cleric	cleric	NOUN	NN	Number=Sing	3	nmod	3:nmod	_
            8	will	will	AUX	MD	VerbForm=Fin	10	aux	10:aux	_
            9	be	be	AUX	VB	VerbForm=Inf	10	aux	10:aux	_
            10	causing	cause	VERB	VBG	VerbForm=Ger	0	root	0:root	_
            11	us	we	PRON	PRP	Case=Acc|Number=Plur|Person=1|PronType=Prs	10	iobj	10:iobj	_
            12	trouble	trouble	NOUN	NN	Number=Sing	10	obj	10:obj	_
            13	for	for	ADP	IN	_	14	case	14:case	_
            14	years	year	NOUN	NNS	Number=Plur	10	obl	10:obl	_
            15	to	to	PART	TO	_	16	mark	16:mark	_
            16	come	come	VERB	VB	VerbForm=Inf	14	acl	14:acl	SpaceAfter=No
            17	.	.	PUNCT	.	_	10	punct	10:punct	SpaceAfter=No
            18	]	]	PUNCT	-RRB-	_	10	punct	10:punct	_
        """
        return data

    @staticmethod
    def read_data(path):
        """
        Reads the data file in a list of strings.
        :param path: str. with the path of the file
        :return: a list of strings with the .conllu format for each sentence
        """
        with open(path, 'r') as file:
            data = file.read() .split('\n\n')

        return data

    @staticmethod
    def parse_conllu(conllu_sent):
        """
        Parses a .conllu format sentence and keep each word with the respective POS tag
        :param conllu_sent: dict with conllu tags
        :return: a list of tuples with each word and its respective POS tag: [('word','POS_tag')]
        """
        sent = list()
        for word in conllu_sent[0][0]:
            sent.append((word['lemma'], word['xpostag']))

        return sent


if __name__ == '__main__':

    path = '/Users/aggelikiromanou/Desktop/MSDS/5_Text_engineering/' \
           'Assignment_3/part-of-speech-tagger/data/en-ud-train.conllu'
    train = DataFetcher.read_data(path)

    data_set = list()
    for sentence in train:
        parsed_data = parse(sentence)
        data_set.append(DataFetcher.parse_conllu(parsed_data))

    print(len(data_set))

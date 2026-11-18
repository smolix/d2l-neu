"""Prompts of the book's question set (lm-book-qa-v1), used by
tools/build_book_qa.py.

Every constant is versioned: a prompt that changes after recording gets a new
name (DRAFT_V2, ...), since the API caches are keyed by the exact request and
the manifest records the sha256 of each prompt. The three few-shot
demonstrations are anchored in `train` documents only (chapter
convolutional-modern, section "Residual Networks", a train document of
corpus r3.1), so no dev or test passage reaches the drafter as an example.

Models (chapter 19 spec, part 1.8 and the author's override of 2026-11-14):
drafter DeepSeek-V4.1-Flash-1017 (DRAFT_V1), checker Qwen3.8-27B (CHECK_V1,
NEIGHBOR_V1), frontier verifier (VERIFY_V1; it replaces the spec's drafter
second check SECOND_V1, which is not used), calibration labellers (CALIB_V1).
"""

TYPE_RULES = {
    'single': (
        'Ask for one specific fact the passage states: a design choice, a '
        'definition, a named method or component, a stated result or '
        'comparison. The answer is a short phrase or one sentence.'),
    'explain': (
        'Ask why or how something works, as the passage explains it (a '
        'reason, a mechanism, a consequence). The answer takes one to three '
        'sentences and restates the passage\'s own explanation.'),
    'numeric': (
        'Ask for a number the passage states in its prose (a count, a size, '
        'a setting, a measured or reported value, a year). The number must '
        'appear in the prose of the passage, not only inside a formula or '
        'a code cell.'),
    'code': (
        'Ask what a code cell of the passage computes, defines, returns or '
        'checks, as visible in its source code or stated in the prose '
        'around it (an argument\'s default, a condition, which layer comes '
        'first, what a function returns). Name the function, class or '
        'variable in backticks so the question stands alone. Never ask for '
        'a printed output or a measured result of running the code.'),
    'multihop': (
        'You get two passages from two different sections. Ask one question '
        'whose answer needs a fact from passage A AND a fact from passage B '
        '(for example: a property defined in one and used in the other, or '
        'a comparison of what the two state). It must read as ONE natural '
        'question about one topic, never two unrelated questions joined by '
        '"and". Give two evidence quotes: the first from passage A, the '
        'second from passage B. If the two passages share no topic that one '
        'such question can combine, reply {"candidates": []}.'),
    'unanswerable': (
        'You get a whole section. Ask questions that a reader of this '
        'section could plausibly ask next, on an adjacent topic, but that '
        'the section does NOT answer (and that a textbook on deep learning '
        'might well not answer either). The reference is "The book does '
        'not answer this." key_facts and evidence are empty lists.'),
}

DRAFT_V1 = {
    'system': (
        'You write exam questions about the textbook Dive into Deep Learning '
        'from passages of the book. Every question you write is later shown '
        'WITHOUT the passage, to a reader or a retrieval system that must '
        'find the answer somewhere in the whole book. Reply with JSON only.'),
    'rules': (
        'Rules for every question:\n'
        '1. Self-contained: it must make sense on its own. Never write "this '
        'section", "the passage", "the text", "above", "below", "the '
        'following code", "the example" or similar. Name the concept, model, '
        'function or experiment the question is about.\n'
        '2. Specific to this book: ask for something this passage states '
        '(a number, a design choice, a stated result, what a code cell '
        'computes, defines or returns as visible in its source or stated in '
        'the prose around it). Never ask for a printed output: the book\'s '
        'printed values change when the notebooks run again.\n'
        '3. Paraphrase: do not copy four or more consecutive words from the '
        'passage into the question, except names of functions, classes, '
        'variables or models.\n'
        '4. One correct answer: the question must have a single answer that '
        'the passage settles; avoid questions with lists of many acceptable '
        'answers.\n'
        '5. reference: the correct answer in one to three sentences. '
        'key_facts: one to three short atomic statements that any correct '
        'answer must contain. evidence: an exact, verbatim quote copied '
        'character for character from the passage (one sentence or one '
        'line of code is best) that supports the answer.\n'
        '6. Write two different candidate questions, on two different facts '
        'when the passage allows it.'),
    'format': (
        'Reply with one JSON object: {"candidates": [{"type": "<type>", '
        '"question": "...", "reference": "...", "key_facts": ["..."], '
        '"evidence": ["<verbatim quote>"]}, {...}]}'),
    'user_single': (
        'Book location: {heading}\n\nPassage:\n<<<\n{text}\n>>>\n\n'
        'Question type: {qtype}. {type_rule}\n\n{rules}\n\n{format}'),
    'user_multihop': (
        'Passage A (book location: {heading_a}):\n<<<\n{text_a}\n>>>\n\n'
        'Passage B (book location: {heading_b}):\n<<<\n{text_b}\n>>>\n\n'
        'Question type: multihop. {type_rule}\n\n{rules}\n\n{format}'),
    'user_unanswerable': (
        'Book location: {heading}\n\nSection:\n<<<\n{text}\n>>>\n\n'
        'Question type: unanswerable. {type_rule}\n\n'
        'Rules: each question is self-contained (never "this section", "the '
        'passage", "above"); specific enough that it would have one answer '
        'if a book stated it; about the subject of this section, not about '
        'unrelated fields. Write two different candidates.\n\n'
        'Reply with one JSON object: {{"candidates": [{{"type": '
        '"unanswerable", "question": "...", "reference": "The book does not '
        'answer this.", "key_facts": [], "evidence": []}}, {{...}}]}}'),
}

# Few-shot demonstrations: hand-written, anchored in the train document
# chapter_convolutional-modern/resnet (corpus r3.1); quotes are verbatim.
FEWSHOT_V1 = [
    {'type': 'single',
     'passage': (
         'ResNet\'s residual block follows VGG\'s design of $3 \\times 3$ '
         'convolutions. It\nhas two $3 \\times 3$ convolutional layers with '
         'the same number of output\nchannels. Each convolution is followed '
         'by batch normalization, and the first\nalso by a ReLU. The '
         'convolutions have no bias, since the batch normalization\nafter '
         'them would remove it (Section "Normalization Layers"). The block '
         'adds its\ninput to the output of the second batch normalization '
         'and applies a final\nReLU.'),
     'answer': {
         'type': 'single',
         'question': ('Why are the convolutional layers in ResNet\'s '
                      'residual block built without a bias term?'),
         'reference': ('Each convolution is followed by batch normalization, '
                       'which would cancel a bias, so the bias is left '
                       'out.'),
         'key_facts': ['batch normalization follows each convolution',
                       'batch normalization would remove the bias'],
         'evidence': ['The convolutions have no bias, since the batch '
                      'normalization\nafter them would remove it']}},
    {'type': 'numeric',
     'passage': (
         'The network starts with a $3 \\times 3$ convolution with 16 output '
         'channels,\nbatch normalization, and a ReLU. Three stages of $n$ '
         'blocks follow, with 16,\n32, and 64 channels; the first block of '
         'the second and of the third stage\nhalves the height and the '
         'width. Global average pooling and a fully\nconnected layer form '
         'the head. With two convolutional layers per block, the\nnetwork '
         'has $6n + 2$ layers: 20 for $n = 3$ and 56 for $n = 9$.'),
     'answer': {
         'type': 'numeric',
         'question': ('In the book\'s Fashion-MNIST comparison of plain and '
                      'residual networks, how many layers does the network '
                      'have when each of its three stages holds nine '
                      'blocks?'),
         'reference': ('It has 56 layers: two convolutional layers per block '
                       'give 6n + 2 layers, and n = 9.'),
         'key_facts': ['56 layers'],
         'evidence': ['20 for $n = 3$ and 56 for $n = 9$']}},
    {'type': 'code',
     'passage': (
         '```python\nclass Residual(nn.Module):  #@save\n    """The Residual '
         'block of ResNet models."""\n    def __init__(self, num_channels, '
         'use_1x1conv=False, strides=1):\n        super().__init__()\n'
         '        # No conv biases: the batch normalization after each conv '
         'removes them\n        self.conv1 = nn.LazyConv2d(num_channels, '
         'kernel_size=3, padding=1,\n                                   '
         'stride=strides, bias=False)\n        self.conv2 = nn.LazyConv2d('
         'num_channels, kernel_size=3, padding=1,\n                         '
         '          bias=False)\n        # Auto-enable 1x1 conv when '
         'downsampling so the residual shape matches.\n        if '
         'use_1x1conv or strides != 1:\n            self.conv3 = '
         'nn.LazyConv2d(num_channels, kernel_size=1,\n                     '
         '                  stride=strides, bias=False)\n            '
         'self.bn3 = nn.LazyBatchNorm2d()\n        else:\n            '
         'self.conv3 = None\n```'),
     'answer': {
         'type': 'code',
         'question': ('In the book\'s `Residual` block class, under what '
                      'condition does the shortcut get a 1x1 convolution '
                      'even when `use_1x1conv` is False?'),
         'reference': ('When `strides` is not 1, that is, when the block '
                       'downsamples; the shortcut then needs the 1x1 '
                       'convolution so its shape matches the branch.'),
         'key_facts': ['when strides is not 1 (the block downsamples)'],
         'evidence': ['if use_1x1conv or strides != 1:']}},
]

CHECK_V1 = {
    'system': (
        'You check exam questions about the textbook Dive into Deep Learning '
        'against the book passage they were written from. Judge from the '
        'passage text alone. Reply with JSON only.'),
    'user': (
        '{passages}\n\nQuestion: {question}\nReference answer: {reference}\n'
        'Key facts:\n{key_facts}\n\n'
        'Answer each with "yes" or "no":\n'
        'self_contained: does the question make sense to a reader who has '
        'NOT seen the passage (it names what it asks about and never points '
        'to "the passage", "this section", "above" or similar)?\n'
        'reference_supported: does the passage text support the reference '
        'answer fully?\n'
        'key_facts_complete: are the key facts correct, and enough to tell a '
        'correct answer from a wrong one?\n'
        'unique_answer: does the question have one correct answer (the '
        'reference), not several equally valid ones?\n'
        'general_knowledge: could a reader with general machine-learning '
        'knowledge but without this book answer it correctly?\n'
        'Also write your own short answer to the question from the '
        'passage.\n\n'
        'Reply with one JSON object: {{"self_contained": "yes|no", '
        '"reference_supported": "yes|no", "key_facts_complete": "yes|no", '
        '"unique_answer": "yes|no", "general_knowledge": "yes|no", '
        '"answer": "..."}}'),
    'user_unanswerable': (
        'Section:\n<<<\n{text}\n>>>\n\nQuestion: {question}\n\n'
        'Answer each with "yes" or "no":\n'
        'self_contained: does the question make sense to a reader who has '
        'NOT seen the section?\n'
        'answered: does the section answer the question, fully or in '
        'part?\n'
        'general_knowledge: could a reader with general machine-learning '
        'knowledge answer it?\n\n'
        'Reply with one JSON object: {{"self_contained": "yes|no", '
        '"answered": "yes|no", "general_knowledge": "yes|no"}}'),
}

NEIGHBOR_V1 = {
    'system': (
        'You check whether other passages of the textbook Dive into Deep '
        'Learning bear on an exam question. Reply with JSON only.'),
    'user': (
        'Question: {question}\nReference answer: {reference}\n\n'
        'Other passages of the book:\n{passages}\n\n'
        'Answer each with "yes" or "no", and list the passage numbers '
        'behind each "yes":\n'
        'contradicts: does any passage contradict the reference answer?\n'
        'answers_better: does any passage answer the question better or '
        'more completely than the reference?\n'
        'answers_equally: does any passage, on its own, answer the question '
        'as well as the reference does?\n\n'
        'Reply with one JSON object: {{"contradicts": "yes|no", '
        '"answers_better": "yes|no", "answers_equally": "yes|no", '
        '"chunk": {{"contradicts": [], "answers_better": [], '
        '"answers_equally": []}}}}'),
    'user_unanswerable': (
        'Question: {question}\n\nPassages of the book:\n{passages}\n\n'
        'answers: does any passage answer the question, fully or in part? '
        '"yes" or "no", with the passage numbers.\n\n'
        'Reply with one JSON object: {{"answers": "yes|no", "chunk": []}}'),
}

VERIFY_V1 = {
    'system': (
        'You verify items of a question set built over the textbook Dive '
        'into Deep Learning. A cheaper model drafted each item from a book '
        'passage; another model checked it. You give an independent, careful '
        'verdict. Reply with JSON only.'),
    'user': (
        'Item type: {qtype}\nQuestion: {question}\nReference answer: '
        '{reference}\nKey facts:\n{key_facts}\n\n'
        'Gold evidence (the span the item is graded against), shown inside '
        'its passage between <<gold>> and <</gold>>:\n{passages}\n'
        '{alternatives}\n'
        'Check, answering "yes" or "no":\n'
        'self_contained: a reader who has not seen the passage understands '
        'what is asked.\n'
        'reference_supported: the gold evidence (with its passage) supports '
        'the reference answer, and the reference is correct.\n'
        'key_facts_complete: the key facts are correct and minimal, and '
        'enough to grade an answer.\n'
        'unique_answer: the question has one correct answer; nothing in the '
        'passages shown, or in what you know about the book, makes another '
        'answer equally right.\n'
        'general_knowledge: a reader with general machine-learning knowledge '
        'but without this book could answer correctly (this does not reject '
        'an item).\n'
        'verdict: "accept" if the first four are all "yes", else "reject".\n'
        'rationale: at most two sentences; for a rejection, say what is '
        'wrong.\n\n'
        'Reply with one JSON object: {{"self_contained": "yes|no", '
        '"reference_supported": "yes|no", "key_facts_complete": "yes|no", '
        '"unique_answer": "yes|no", "general_knowledge": "yes|no", '
        '"verdict": "accept|reject", "rationale": "..."}}'),
    'user_unanswerable': (
        'Item type: unanswerable (the book should NOT answer it).\n'
        'Question: {question}\n\n'
        'The section it was drafted from:\n<<<\n{text}\n>>>\n\n'
        'The ten passages of the whole book that a hybrid retriever ranks '
        'highest for the question:\n{passages}\n\n'
        'Check, answering "yes" or "no":\n'
        'self_contained: a reader understands what is asked without the '
        'section.\n'
        'answered_by_book: the section or any of the passages answers the '
        'question, fully or in part.\n'
        'general_knowledge: a reader with general machine-learning knowledge '
        'could answer it.\n'
        'verdict: "accept" if self_contained is "yes" and answered_by_book '
        'is "no", else "reject".\n'
        'rationale: at most two sentences.\n\n'
        'Reply with one JSON object: {{"self_contained": "yes|no", '
        '"answered_by_book": "yes|no", "general_knowledge": "yes|no", '
        '"verdict": "accept|reject", "rationale": "..."}}'),
}

CALIB_V1 = {
    'system': (
        'You label answers of a retrieval-augmented question-answering '
        'system over the textbook Dive into Deep Learning. Your labels '
        'calibrate a small automatic judge. Reply with JSON only.'),
    'user': (
        'Question: {question}\nReference answer: {reference}\n'
        'Key facts:\n{key_facts}\n\nPassages given to the system:\n'
        '{passages}\n\nThe system\'s answer:\n{answer}\n\n'
        'Label:\n'
        'correct: true if the answer states every key fact (in any wording) '
        'and nothing contradicting the reference; for a question the book '
        'does not answer (reference "The book does not answer this."), true '
        'only if the system declined to answer.\n'
        'faithful: true if every factual statement of the answer is '
        'supported by the passages given (ignore the reference here); an '
        'answer that declines is faithful.\n'
        'rationale: one sentence.\n\n'
        'Reply with one JSON object: {{"correct": true|false, "faithful": '
        'true|false, "rationale": "..."}}'),
}

PROMPTS = {'DRAFT_V1': DRAFT_V1, 'TYPE_RULES': TYPE_RULES,
           'FEWSHOT_V1': FEWSHOT_V1, 'CHECK_V1': CHECK_V1,
           'NEIGHBOR_V1': NEIGHBOR_V1, 'VERIFY_V1': VERIFY_V1,
           'CALIB_V1': CALIB_V1}

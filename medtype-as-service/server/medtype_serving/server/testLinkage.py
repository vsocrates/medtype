import argparse

from entity_linkers import * 
from MedTypeServerEdits import *
from helper import check_max_seq_len

parser = argparse.ArgumentParser(description='Start a MedTypeServer for serving')

group2 = parser.add_argument_group('MedType Parameters', 'config how MedType model and pooling works')
group2.add_argument('-max_seq_len', 		type=check_max_seq_len, default=25, help='maximum length of a sequence, longer sequence will be trimmed on the right side. '
                                                'set it to NONE for dynamically using the longest sequence in a (mini)batch.')
group2.add_argument('-cased_tokenization', 	dest='do_lower_case', 	action='store_false', default=True, help='Whether tokenizer should skip the default lowercasing and accent removal.'
                                                        'Should be used for e.g. the multilingual cased pretrained MedType model.')


group1 = parser.add_argument_group('File Paths', 'config the path, checkpoint and filename of a pretrained/fine-tuned MedType model')
group1.add_argument('--model_path', 		type=str, 	required=True, 			help='directory of a pretrained MedType model')
group1.add_argument('--model_type', 		type=str, 	default='bert_combined', 	help='Model type. Options: [bert_combined (Bio arcticles/EHR), bert_plain (General)]')
group1.add_argument('--entity_linker', 		type=str, 	default='scispacy',		help='entity linker to use over which MedType is employed. Options: \
                                                    [scispacy, qumls, ctakes, metamap, metamaplite]')
group1.add_argument('--dropout', 		type=float, 	default=0.1, 			help='Dropout in MedType Model.')

group1.add_argument('--tokenizer_model', 	type=str, 	default='bert-base-cased',	help='tokenizer to use')
group1.add_argument('--context_len', 		type=int, 	default=120,			help='Number of tokens to consider in context for predicting mention semantic type')
group1.add_argument('--model_batch_size', 	type=int, 	default=256, 			help='maximum number of mentions handled by MedType model')
group1.add_argument('--threshold', 		type=float, 	default=0.5, 			help='Threshold used on logits from MedType')
group1.add_argument('--type_remap_json', 	type=str, 	required=True, 			help='Json file containing semantic type remapping to coarse-grained types')
group1.add_argument('--type2id_json', 		type=str, 	required=True, 			help='Json file containing semantic type to identifier mapping')
group1.add_argument('--umls2type_file', 	type=str, 	required=True, 			help='Location where UMLS to semantic types mapping is stored')

# Entity Linkers Arguments
# group1.add_argument('--quickumls_path',		type=str, 	default=None, 			help='Location where QuickUMLS is installed')
# group1.add_argument('--metamap_path', 		type=str, 	default=None, 			help='Location where MetaMap executable is installed, e.g .../public_mm/bin/metamap18')
# group1.add_argument('--metamaplite_path', 	type=str, 	default=None, 			help='Location where MetaMapLite is installed, e.g .../public_mm_lite')

parser.add_argument('--verbose', 		action='store_true', 	default=False, 	help='turn on tensorflow logging for debug')

args = parser.parse_args()


medtype = MedTypeWorkers(args)


test_texts = [
    "Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis (MTB) bacteria.",
    "Peripheral Vascular: (Right radial pulse: Not assessed), (Left radial pulse: Not assessed), (Right DP pulse: Not assessed), (Left DP pulse: Not assessed) Respiratory / Chest: (Breath Sounds: Clear : ) Abdominal: Soft, Non-tender, Bowel sounds present Extremities: Right: Absent, Left: Absent",
    "Communication: Patient discussed on interdisciplinary rounds. Discussed in detail with Dr. [**Last Name (STitle) **], with vascular access team, and with blood bank team.",
    "Patient was reportedly intubated for unclear reasons and transferred to [**Hospital1 19**] on [**2111-4-14**] for further management. Imaging here did not reveal hemorrhage. Altered mental status was thought [**3-17**] UTI c/w enteroccocus. Patient was extubated on [**2111-4-15**]."
]

message = {
    "entity_linker" : "scispacy",
    "text" : test_texts
}

filtered = medtype.run(message)
print(filtered)
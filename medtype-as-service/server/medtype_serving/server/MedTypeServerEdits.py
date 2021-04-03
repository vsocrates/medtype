#!/usr/bin/env python

import multiprocessing, threading, os, random, sys, json, time, numpy as np, pickle

from itertools import chain
from datetime import datetime
from multiprocessing import Process
from multiprocessing.pool import Pool
from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import OrderedDict
from collections import defaultdict as ddict

import torch
# from termcolor import colored

from helper import *

from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from medtype.models import BertCombined, BertPlain
from entity_linkers import ScispaCy, QUMLS, Metamap, MetamapLite

__version__ = '1.0.0'

class ServerCmd:
	terminate	= b'TERMINATION'
	show_config	= b'SHOW_CONFIG'
	show_status	= b'SHOW_STATUS'
	new_job		= b'REGISTER'
	elink_out	= b'ELINKS'

	@staticmethod
	def is_valid(cmd):
		return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())

class MedTypeWorkers(Process):

	def __init__(self, args):
		super().__init__()
		self.args 			= args
		self.max_seq_len		= args.max_seq_len
		self.do_lower_case		= args.do_lower_case
		self.model_path			= args.model_path
		self.model_type			= args.model_type
		self.model_params		= self.load_model(args.model_path)
		self.entity_linker 		= args.entity_linker
		self.dropout 			= args.dropout
		self.verbose			= args.verbose
		self.tokenizer_model 		= args.tokenizer_model
		self.context_len 		= args.context_len
		self.batch_size 		= args.model_batch_size
		self.threshold 			= args.threshold
		self.ent_linker 		= self.get_linkers(args.entity_linker)
		self.type_remap		= json.load(open(args.type_remap_json))
		self.type2id		= json.load(open(args.type2id_json))
		self.umls2type		= pickle.load(open(args.umls2type_file, 'rb'))
		self.id2type			= {v: k for k, v in self.type2id.items()}

		self.tokenizer	= BertTokenizer.from_pretrained(self.tokenizer_model)
		self.tokenizer.add_tokens(['[MENTION]', '[/MENTION]'])
		self.cls_tok	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]'))
		self.sep_tok	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[SEP]'))
		self.men_start  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[MENTION]'))
		self.men_end 	= self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[/MENTION]'))


	def load_model(self, model_path):
		state 		= torch.load(model_path, map_location="cpu")
		state_dict	= state['state_dict']
		new_state_dict	= OrderedDict()

		for k, v in state_dict.items():
			if 'module' in k:
				k = k.replace('module.', '')
			new_state_dict[k] = v

		return new_state_dict

	
	def get_ent_linker(self, linker):
		if   linker.lower() == 'scispacy': 	return ScispaCy(self.args)
		elif linker.lower() == 'quickumls': 	return QUMLS(self.args)
		elif linker.lower() == 'metamap': 	return Metamap(self.args)
		elif linker.lower() == 'metamaplite': 	return MetamapLite(self.args)
		else: raise NotImplementedError

	def get_linkers(self, linkers_list):
		ent_linkers = {}
		for linker in linkers_list.split(','):
			ent_linkers[linker] = self.get_ent_linker(linker)
		return ent_linkers

	def pad_data(self, data):
		max_len  = np.max([len(x['toks']) for x in data])
		tok_pad	 = np.zeros((len(data), max_len), np.int32)
		tok_mask = np.zeros((len(data), max_len), np.float32)
		men_pos  = np.zeros((len(data)), np.int32)
		meta 	 = []

		for i, ele in enumerate(data):
			tok_pad[i, :len(ele['toks'])]   = ele['toks']
			tok_mask[i, :len(ele['toks'])]  = 1.0
			men_pos[i] 			= ele['men_pos'] 
			meta.append({
				'text_id': ele['text_id'],
				'men_id' : ele['men_id']
			})

		return torch.LongTensor(tok_pad).to(self.device), torch.FloatTensor(tok_mask).to(self.device), torch.LongTensor(men_pos).to(self.device), meta

	def get_batches(self, elinks):
		data_list = []

		for t_id, ele in enumerate(elinks):
			text = ele['text']

			for m_id, men in enumerate(ele['mentions']):
				start, end = men['start_offset'], men['end_offset']

				mention   = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[start:end]))
				prev_toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[:start]))[-self.context_len//2:]
				next_toks = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[end:]))[:self.context_len//2]
				toks	  = self.cls_tok + prev_toks + self.men_start + mention + self.men_end + next_toks + self.sep_tok

				data_list.append({
					'text_id'	: t_id,
					'men_id'	: m_id,
					'toks'		: toks,
					'men_pos'	: len(prev_toks) + 1
				})

		num_batches = int(np.ceil(len(data_list) / self.batch_size))
		for i in range(num_batches):
			start_idx = i * self.batch_size
			yield self.pad_data(data_list[start_idx : start_idx + self.batch_size])

	def filter_candidates(self, elinks):
		out = ddict(lambda: ddict(dict))

		with torch.no_grad():

			for batch in self.get_batches(elinks):

				logits = self.model(
						input_ids	= batch[0], 
						attention_mask 	= batch[1],
						mention_pos_idx = batch[2]
					)

				preds = (torch.sigmoid(logits) > self.threshold).cpu().numpy()

				for i, ele in enumerate(batch[3]):
					out[ele['text_id']][ele['men_id']] = set([self.id2type[x] for x in np.where(preds[i])[0]])

		filt_elinks = []
		for t_id, ele in enumerate(elinks):
			mentions = []
			for m_id, men in enumerate(ele['mentions']):
				men['pred_type'] = out[t_id][m_id]

				# No filtering when predicted type is NA
				if len(men['pred_type']) == 0:  
					men['filtered_candidates'] = men['candidates']
				else: 				
					men['filtered_candidates'] = [[cui, scr] for cui, scr in men['candidates'] if len(self.umls2type.get(cui, set()) & men['pred_type']) != 0]
					if len(men['filtered_candidates']) == 0:
						men['filtered_candidates'] = men['candidates']

				men['pred_type'] = list(men['pred_type'])	# set is not JSON serializable
				mentions.append(men)

			ele['mentions'] = mentions
			filt_elinks.append(ele)

		return filt_elinks

	def run(self, message):
		'''
		Runs the entity typing algorithm on a message

		Args: 
			message : dict of shape {'entity_linker' : name of linker,
									'text' : [list of texts] }
		'''
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		if   self.model_type.lower() == 'bert_combined': self.model = BertCombined(len(self.tokenizer), len(self.type2id), self.dropout)
		elif self.model_type.lower() == 'bert_plain':    self.model = BertPlain(len(self.tokenizer), len(self.type2id), self.dropout)
		else: raise NotImplementedError

		self.model.to(self.device)
		self.model.load_state_dict(self.model_params, strict=False)
            
		if message['entity_linker'] in self.ent_linker:
			elinks = []
			for text in message['text']:
				elinks.append(self.ent_linker[message['entity_linker']](text))
			filt_elinks = self.filter_candidates(elinks)

		return filt_elinks
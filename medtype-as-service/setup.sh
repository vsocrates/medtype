DEFAULT='\033[0m'
BOLD='\033[1;32m\e[1m'

if ! python -c "import gdown" &> /dev/null; then
	echo -e "${BOLD} MedType> Install gdown package ${DEFAULT}"
	pip install gdown
fi

if [ ! -d "./resources/pretrained_models" ]
then
	echo -e "${BOLD} MedType> Setting up directories ${DEFAULT}"
	mkdir -p resources/pretrained_models
fi

if [ ! -f "./resources/pretrained_models/general_model.bin" ]
then
	# Pretrained model for General domain text
	echo -e "${BOLD} MedType> Downloading pre-trained model for general articles: general_model.bin ${DEFAULT}"
	gdown --id 15vKHwzEa_jcipyEDClNSzJguPxk0VOC7 -O resources/pretrained_models/general_model.zip

	echo -e "${BOLD} MedType> Extracting model ${DEFAULT}"
	unzip resources/pretrained_models/general_model.zip -d resources/pretrained_models/
	rm -f resources/pretrained_models/general_model.zip
fi

if [ ! -f "./resources/pretrained_models/pubmed_model.bin" ]
then
	# Pretrained model for Biomedical research articles
	echo -e "${BOLD} MedType> Downloading pre-trained model for biomedical articles: pubmed_model.bin ${DEFAULT}"
	gdown --id 1So-FMFyPMup84VvbWqH7Cars8jfjEIx_ -O resources/pretrained_models/pubmed_model.zip

	echo -e "${BOLD} MedType> Extracting model ${DEFAULT}"
	unzip resources/pretrained_models/pubmed_model.zip -d resources/pretrained_models/
	rm -f resources/pretrained_models/pubmed_model.zip
fi


if [ ! -f "./resources/pretrained_models/ehr_model.bin" ]
then
	# Pretrained model for EHR documents
	echo -e "${BOLD} MedType> Downloading pre-trained model for Electronic Health Records (EHRs): ehr_model.bin ${DEFAULT}"
	gdown --id 1t2QlpEWnHOMdts4h3y55hVA9Wh2ZbjKi -O resources/pretrained_models/ehr_model.zip

	echo -e "${BOLD} MedType> Extracting model ${DEFAULT}"
	unzip resources/pretrained_models/ehr_model.zip -d resources/pretrained_models/
	rm -f resources/pretrained_models/ehr_model.zip
fi

if [ ! -f "./resources/umls2type.pkl" ]
then
	echo -e "${BOLD} MedType> Downloading UMLS to Semantic Type Mapping ${DEFAULT}"
	gdown --id 1Jly06IjI7iWgQRmj456filYD5FyBLQAg -O resources/umls2type.zip
	unzip resources/umls2type.zip -d resources/
	rm -f resources/umls2type.zip
fi 

# echo -e "${BOLD} MedType> Installing Server and Client Packages ${DEFAULT}"
# cd server
# python setup.py clean
# python setup.py install


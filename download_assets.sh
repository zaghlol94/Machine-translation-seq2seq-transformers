python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
gdown "1UVVJ6Cra5F0JjMLTy3vRBXEwLimjS6jT"
mv transformers.zip src/
cd src/
unzip transformers.zip
rm transformers.zip

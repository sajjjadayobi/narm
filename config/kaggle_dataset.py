!pip install --upgrade --force-reinstall --no-deps kaggle
# got to kaggle account and create new API token then download kaggle.json
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
# you must accept relus and late submistion on this competition
!kaggle competitions download -c COMPETITION_NAME

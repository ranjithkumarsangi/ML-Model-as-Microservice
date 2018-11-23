import pickle
import pandas as pd
import numpy as np
from rest_framework import views
from rest_framework.response import Response
from .serializers import PredictionSerializer
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema



class Prediction(views.APIView):
	
	data_param = openapi.Parameter('data', in_=openapi.IN_QUERY, description='data',type=openapi.TYPE_STRING)	
	
	@staticmethod
	def read_dataset(file_url):
		df = pd.read_csv(file_url)
		categorical_columns = ['job','marital','education','default','housing','loan','contact','month','poutcome']
		df_new  = pd.get_dummies(df,columns=categorical_columns)
		return df_new.drop(['deposit'],axis=1)
		
		
	@swagger_auto_schema(manual_parameters=[data_param])
	def get(self,request):
		df = self.read_dataset('bank.csv')
		df = df.loc[0:0]
		
		for i, t in enumerate(df.dtypes):
			if t == object:
				df.iat[0,i] = ''
			else:
				df.iat[0,i] = 0
		data = request.query_params.get('data',None)
		estimator = pickle.load(open('classifier.pkl','rb'))
		features = [x.strip() for x in data.split(',')]
		
		df.at[0,'age'] = features[0]
		df.at[0,'job_'+features[1]] = 1
		df.at[0,'marital_'+features[2]] = 1
		df.at[0,'education_'+features[3]] = 1
		df.at[0,'default_'+features[4]] = 1
		df.at[0,'balance'] = features[5] 
		df.at[0,'housing_'+features[6]] = 1
		df.at[0,'loan_'+features[7]] = 1
		df.at[0,'contact_'+features[8]] = 1
		df.at[0,'day'] = features[9]
		df.at[0,'month_'+features[10]] = 1
		df.at[0,'duration'] = features[11]
		df.at[0,'pdays'] = features[12]
		df.at[0,'previous'] = features[13]
		df.at[0,'poutcome_'+features[14]] = 1
		
			
		if hasattr(estimator,'predict_proba'):
			result = estimator.predict_proba(df)
		else:
			result = [[0.0,0.0]]
			result[0][int(estimator.predict(df))] = 1.0
		predictions = []
		for i,p in np.ndenumerate(result[0]):
			predictions.append({
			'prediction':i[0],
			'prob':p
			})
		response = PredictionSerializer(predictions,many=True).data
		return Response(response)



from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
	prediction 	= serializers.IntegerField()
	prob = serializers.FloatField()
	
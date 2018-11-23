"""ml_rest URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from django.conf.urls import url
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

schema_view = get_schema_view(
	openapi.Info(
		title = "My Machine Learning API",
		default_version = 'v1',
		description = "ML Problem Statement:Predicting Term Deposit Suscriptions.\n Author:Ranjith Kumar Sangi",
		),
		public = True,
		permission_classes = (permissions.AllowAny,),
	)

urlpatterns = [
    #path('admin/', admin.site.urls),
	path('api/v1/',include('prediction.urls')),
	url(r'^swagger(?P<format>\.json|\.yaml)$',schema_view.without_ui(cache_timeout=None),name='schema-json'),
	url(r'^swagger/$',schema_view.with_ui('swagger',cache_timeout=None),name='schema-swagger-ui'),
	url(r'^redoc/$',schema_view.with_ui('redoc',cache_timeout=None),name='schema-redoc'),
]
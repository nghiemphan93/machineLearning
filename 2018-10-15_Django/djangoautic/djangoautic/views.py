from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpRequest

def home(request: HttpRequest):
   #return HttpResponse("home")
   return render(request, "home.html")


def about(request: HttpRequest):
   return HttpResponse("about")


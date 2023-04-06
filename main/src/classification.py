import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.nn import Softmax

def scale_ratings(rating, min_rating, max_rating):
  
    probability = (rating - min_rating) / (max_rating - min_rating)
    return probability


def gender(model, extractor, df, root_dir):
  prompts = ['a face', 'a drawing']
  softmax = Softmax(dim=1)
  results = []
  for name in df['name'].str.lower():
    
    for prompt in prompts:
      
      for i in range(50):
        
        path = os.path.join(root_dir, name, prompt, f'{name}_{i}.jpeg')
        img = Image.open(path)
        inputs = extractor(img, return_tensors = 'pt')
        
        with torch.no_grad():
          logits = model(**inputs).logits #compute logits
          prob_scores = softmax(logits)[0] #convert logits to probabilities
          female_prob = prob_scores[0].item() #probability score for female class
          male_prob = prob_scores[1].item() #probability score for male class
          


          result = {
            'name': name,
            'image_idx': i,
            'prompt': prompt,
            'female_prob': female_prob,
            'male_prob': male_prob            
          }

          results.append(result)
          
  return pd.DataFrame.from_dict(results)


def emotion(model, extractor, df, root_dir):
  prompts = ['a face', 'a drawing']
  softmax = Softmax(dim=1)
  results = []
  for name in df['name'].str.lower():
    
    for prompt in prompts:
      
      for i in range(50):
        
        path = os.path.join(root_dir, name, prompt, f'{name}_{i}.jpeg')
        img = Image.open(path)
        inputs = extractor(img, return_tensors = 'pt')
        
        with torch.no_grad():
          logits = model(**inputs).logits #compute logits
          prob_scores = softmax(logits)[0] #convert logits to probabilities
          angry_prob = prob_scores[0].item() #probability score for angry class
          disgust_prob = prob_scores[1].item()
          fear_prob = prob_scores[2].item()
          happy_prob = prob_scores[3].item()
          neutral_prob = prob_scores[4].item()
          sad_prob = prob_scores[5].item()
          surprise_prob = prob_scores[6].item()          

          result = {
            'name': name,
            'image_idx': i,
            'prompt': prompt,
            'angry_prob': angry_prob,
            'disgust_prob': disgust_prob,
            'fear_prob': fear_prob,
            'happy_prob': happy_prob,
            'neutral_prob': neutral_prob,
            'sad_prob': sad_prob,
            'surprise_prob': surprise_prob            
          }

          results.append(result)
          
  return pd.DataFrame.from_dict(results)

def age(model, extractor, df, root_dir):
  prompts = ['a face', 'a drawing']
  softmax = Softmax(dim=1)
  results = []
  for name in df['name'].str.lower():
    
    for prompt in prompts:
      
      for i in range(50):
        
        path = os.path.join(root_dir, name, prompt, f'{name}_{i}.jpeg')
        img = Image.open(path)
        inputs = extractor(img, return_tensors = 'pt')
        
        with torch.no_grad():
          logits = model(**inputs).logits #compute logits
          prob_scores = softmax(logits)[0] #convert logits to probabilities
          young_prob = prob_scores[0:4].sum().item() #probability score for angry class
          old_prob = prob_scores[6:].sum().item()

          result = {
            'name': name,
            'image_idx': i,
            'prompt': prompt,
            'young_prob': young_prob,
            'old_prob': old_prob
          }

        

          results.append(result)
          
  return pd.DataFrame.from_dict(results)




          

  
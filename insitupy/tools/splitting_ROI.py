import insitupy
from insitupy._core.data import InSituData,ImageData, CACHE
#from insitupy import InSituExperiment
from insitupy.io import read_xenium
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from anndata import AnnData
import math
import json




def splitting_exp_in_ROI(exp,side_length,image_key,layer):
    
    shape_image=exp.images[image_key][0].shape
    factor=exp.images.metadata[image_key]["pixel_size"]
    height=shape_image[0]*factor
    width=shape_image[1]*factor
    
    number_of_ROI=int(math.ceil(width / side_length)*math.ceil(height / side_length))
  
    ROI={}
    datasets={}
    x=0
    y=0
   
    while x < width or y < height:
        if (x+side_length <= width) & (y+side_length <= height):
            xlim=(x,x+side_length)
            ylim=(y,y+side_length)
        if (x+side_length > width) & (y+side_length > height):
            xlim=(x,width)
            ylim=(y,height)   
        if (x+side_length > width) & (y+side_length <= height):
            xlim=(x,width)
            ylim=(y,y+side_length)  
        if (x+side_length <= width) & (y+side_length > height):
            xlim=(x,x+side_length)
            ylim=(y,height)   
        
        shape=(xlim[1],ylim[1])
          
        try:
            exp_cropped = exp.crop(xlim=xlim, ylim=ylim)
            datasets[shape]=exp_cropped
            print(shape,":",f"{exp_cropped.cells[layer].matrix.shape[0]} cells identified!")
            df={}
            df['ylim']=ylim
            df['xlim']=xlim
            ROI[shape]=df
        except ValueError:
            print(shape,":","no cells in the this region!")
            #exp_cropped = anndata.AnnData(
            #X=np.empty((0, exp.cells[layer].matrix.n_vars)),         
            #obs=pd.DataFrame(index=[]),          
            #var=exp.cells[layer].matrix.var.copy())
        
        x =xlim[1]
        y =ylim[1] 
        
        y_value=ylim[1]    
        while y_value < height:
            xlim_neu=xlim
            if y_value+side_length < height:
                ylim_neu=(y_value,y_value+side_length)
            else:
                ylim_neu=(y_value,height)
            
            shape=(xlim_neu[1],ylim_neu[1])
            try:
                exp_cropped = exp.crop(xlim=xlim_neu, ylim=ylim_neu)
                datasets[shape]=exp_cropped
                print(shape,":",f"{exp_cropped.cells[layer].matrix.shape[0]} cells identified!")
                df={}
                df['ylim']=ylim_neu
                df['xlim']=xlim_neu
                ROI[shape]=df
            except ValueError:
                print(shape,":","no cells in the this region!")
                #exp_cropped = anndata.AnnData(
                #X=np.empty((0, exp.cells[layer].matrix.n_vars)),         
                #obs=pd.DataFrame(index=[]),          
                #var=exp.cells[layer].matrix.var.copy())
            
        
            y_value=ylim_neu[1]
           
    
        x_value=xlim[1]
        while x_value < width:
            ylim_neu=ylim
            if x_value+side_length < width:
                xlim_neu=(x_value,x_value+side_length)
            else:
                xlim_neu=(x_value,width)
                
            shape=(xlim_neu[1],ylim_neu[1])
            
            try:
                exp_cropped = exp.crop(xlim=xlim_neu, ylim=ylim_neu)
                datasets[shape]=exp_cropped
                print(shape,":",f"{exp_cropped.cells[layer].matrix.shape[0]} cells identified!")
                df={}
                df['ylim']=ylim_neu
                df['xlim']=xlim_neu
                ROI[shape]=df
            except ValueError:
                print(shape,":","no cells in the this region!")
                #exp_cropped = anndata.AnnData(
                #X=np.empty((0, exp.cells[layer].matrix.n_vars)),         
                #obs=pd.DataFrame(index=[]),          
                #var=exp.cells[layer].matrix.var.copy())
            
            x_value=xlim_neu[1]       
                    
    return ROI,datasets

import json
import os

def generate_ROI_geojson(ROI, factor, output_directory):
    feature_list = []

    for idx, region in ROI.items():
        x_min = region['xlim'][0] / factor
        x_max = region['xlim'][1] / factor
        y_min = region['ylim'][0] / factor
        y_max = region['ylim'][1] / factor

        feature = {
            "type": "Feature",
            "id": f"roi_{idx}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                   # [x_min, y_min] 
                ]]
            },
            "properties": {
                "name": f"ROI_{region['xlim'][1]}_{region['ylim'][1]}"
            }
        }

        feature_list.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": feature_list
    }

    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "ROI.geojson"), "w") as f:
        json.dump(geojson_data, f, indent=2)

    

        

def ROI_neighborhood_definition(exp,datasets, side_length,image_key):
    
    shape_image=exp.images[image_key][0].shape
    factor=exp.images.metadata[image_key]["pixel_size"]
    height=shape_image[0]*factor
    width=shape_image[1]*factor
    
    
    shapes = list(datasets.keys())
    dict_neighbors={}
    for idx, data in datasets.items():
        
        x_max=idx[0]
        y_max=idx[1]
        
        
        lh_x= x_max - side_length   # left, high corner (x-max value)
        lh_y=y_max + side_length    # left, high corner (y-max value)
        lh=(lh_x,lh_y)
        
        mh_x= x_max                 # middle, high corner (x-max value)
        mh_y= y_max + side_length   # middle, high corner (y-max value)
        mh=(mh_x,mh_y)
        
        rh_x= x_max + side_length   # right, high corner (x-max value)
        rh_y= y_max + side_length   # right, high corner (y-max value)
        rh=(rh_x,rh_y)
        
        lm_x= x_max - side_length   # left, middle corner (x-max value)
        lm_y= y_max                 # left, middle corner (y-max value)
        lm=(lm_x,lm_y)
        
        rm_x=x_max + side_length    # right, middle corner (x-max value)
        rm_y= y_max                 # right, middle corner (y-max value)
        rm=(rm_x,rm_y)
        
        lb_x= x_max - side_length   # left, bottom corner (x-max value)
        lb_y= y_max - side_length   # left, bottom corner (y-max value)
        lb=(lb_x,lb_y)
        
        mb_x= x_max                 # middle, bottom corner (x-max value)
        mb_y= y_max - side_length   # middle, bottom corner (y-max value)
        mb=(mb_x,mb_y)
        
        rb_x= x_max + side_length   # right, bottom corner (x-max value)
        rb_y= y_max - side_length   # right, bottom corner (y-max value)
        rb=(rb_x,rb_y)
        
        
        lh, mh, rh, lm, rm, lb, mb, rb = [list(p) for p in [lh, mh, rh, lm, rm, lb, mb, rb]]
        
        for n in [lh, mh, rh, lm, rm, lb, mb, rb]:
            n[0] = min(n[0], width)
            n[1] = min(n[1], height)
            
        lh, mh, rh, lm, rm, lb, mb, rb = [tuple(n) for n in [lh, mh, rh, lm, rm, lb, mb, rb]]

        
        number_of_neighbors = sum(1 for n in [lh, mh, rh, lm, rm, lb, mb, rb] if n in shapes)
        if number_of_neighbors==8:
            dict_neighbors[idx] = 'middle'
        else:
            dict_neighbors[idx] = "edge"
   
    return dict_neighbors
#from time import time
#import matplotlib.animation
#import random
#import attack
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.pyplot as plt
import numpy as np


class animated : 
    def __init__(self, G):
        
        self.init_color = "royalblue"
        self.init_edge = "black"
        self.pushed_color = "red"
        self.resist_color = "gainsboro"
        self.init1_colors = 'gray'
        self.used_colors = 'coral'
        self.hospot_colors = 'black'
        self.init_width  = 1
        self.resist_width = 0.5
        self.used_width  = 2
        self.hosp_width = 3
        self.def_colors = 'green'
        self.def_widths = 3
        self.colors = [self.init_color for n in G.nodes]
        self.colors_edge = [self.init_edge for n in G.edges]
        self.clignet = [self.init_color for n in G.nodes][:]
        
        self.widths = [self.init_width for e in G.edges]
        self.animation = []
        self.animation = {'colors':[] ,'widths':[] ,'colors_edge':[],'clignet':[]}
        self.current = None
        self.graph = G
        
        
    def append_animation(self):
        
        colors = self.colors[:]
        colors_edge = self.colors_edge[:]
        if self.current != None:   
            colors[self.current] = self.init_color 
            colors_edge[self.current]  = self.init_edge
            
        self.animation['colors'].append(colors)
        self.animation['colors_edge'].append(colors_edge)       
        self.animation['widths'].append(self.widths[:]) 
       
    
    def push(self,data):
        for n in data:
            self.colors[n] = self.pushed_color
            
        self.append_animation()
    
    
    def push_edges(self,liste,hosp):
        
        for n in liste:
            if n not in hosp:
                self.widths[n] = self.used_width
                self.colors_edge[n] = self.used_colors
            else:
                self.widths[n] = self.hosp_width
                self.colors_edge[n] = self.def_colors
            
        self.append_animation()       
    
    def push_inv_edges(self,liste,hosp):
        for n in liste:
            self.widths[n] = self.init_width
            self.colors_edge[n] = self.init_edge  
            
        for j in hosp:
            self.widths[j] = self.hosp_width
            self.colors_edge[j] = self.hospot_colors 
        self.append_animation()       
    
    
    def push_resist(self,data,list_edge,hosp):
        for n in data:
            self.colors[n] = self.resist_color 
            
        for i in list_edge:
            
            if i not in hosp:
                self.colors_edge[i] = self.resist_color
                self.widths[i] = self.init_width
           
                
    
        
    def suscep_resist(self, liste_edge,liste_resist):
        
        for n in liste_edge:
            if n not in liste_resist:
                self.colors_edge[n] = self.init_edge
                self.widths[n] = self.init_width
            else :
                self.colors_edge[n] = self.resist_color
                self.widths[n] = self.init_width
        self.append_animation() 
        
                
    def hospot_anime(self,hosp):
        
        for n in hosp:
                self.widths[n] = self.hosp_width
                self.colors_edge[n] = self.hospot_colors 
         
        self.append_animation() 
                
          
    def push_defence(self,data,liste_edge,edge_def,honeypot):
        
        for n in data:
            self.colors[n] = self.init_color
            
        #for i in liste_edge : 
        #    self.colors_edge[i]= self.init_edge
        #    self.widths[i] = self.used_width
            
        for j in edge_def:
            if j not in honeypot:
                self.colors_edge[j]= self.init_edge
                self.widths[j] = self.init_width
        for n in honeypot:
            self.widths[n] = self.hosp_width
            self.colors_edge[n] = self.hospot_colors
        self.append_animation() 
        
    #def push_defence_inv(self,data,liste_edge):
        
    #    for n in data:
    #        self.colors[n] = self.used_colors
            
    #    for i in liste_edge : 
    #        self.colors_edge[i]= self.used_colors
    #        self.widths[i] = self.used_width
            
        
    #    self.append_animation() 
                
    
    def push_res(self,data):
        for n in data:
            self.colors[n] = self.resist_color
        
        self.append_animation() 
        
    def push_def(self,liste_edges,hosp):
        
        for i in liste_edges : 
            self.colors_edge[i]= self.def_colors
            self.widths[i] = self.def_widths
        for n in hosp:
            self.widths[n] = self.hosp_width
            self.colors_edge[n] = self.hospot_colors
        self.append_animation() 
        
     #def push_def(self,list):
    #    for n in list:
    #        self.colors[n] = self.init_color
               
            
    def push_IS(self,data,edge,hosp):
        
        for n in data: 
                
            self.colors[n] = self.init_color
            
        for i in edge :
            if i not in hosp: 
                self.colors_edge[i]= self.init_edge
                self.widths[i] = self.init_width        
        
               
        self.append_animation()
    
  
        
        

class queue(animated):
    
    def __init__(self, G):
        
        super().__init__(G)
        #les données sont stockées dans un tableau
        self.data = []
        self.table =[]
        self.graphe =[]
        self.cumule = []
        self.edge =[]
        self.table1 = []
        self.susp = []
        self.resist =[]
        self.compt =[]
        self.new =[]
        
    #def compteur(self,compt):
     #   self.compt = compt
        
  
    def push (self, state = None): 
        li = []
        resist = []
        for n in range(len(state)): 
            if state[n]==1:  
                if  self.graph.nodes[n]['index'] not in li:
                    li.append(self.graph.nodes[n]['index'])
                    #print("li================",li)  
            elif state[n]== -1:
                if self.graph.nodes[n]['index'] not in resist:
                    resist.append(self.graph.nodes[n]['index'])
        super().push(li) 
        super().push_res(resist)
        self.data.extend(li)
        self.data = list( dict.fromkeys(self.data) )
        
    def push_edges(self,edges,hosp):
        for n in edges:
            self.edge.append(n)
        
        super().push_edges(edges,hosp)
        
        
    def push_def(self,liste_edges):
        super().push_def(liste_edges)
     
        
           
    def push_inv_edges(self,edges,hosp):
        for n in edges:
            self.edge.append(n)
        
        super().push_inv_edges(edges,hosp)
           
        
    def push_IS(self,node_IS ,edge_IS,hosp):
        nodes = []
        
        for n in node_IS:
            nodes.append(self.graph.nodes[n]['index'])
            
        super().push_IS(nodes,edge_IS,hosp)
                
                         
    def push_resist(self,state,list_edge,hosp):
        list = []
      
        for n in range(len(state)):
            
            if state[n]==-1:
                if  self.graph.nodes[n]['index'] not in list:
                    list.append(self.graph.nodes[n]['index'])
                    
                super().push_resist(list,list_edge,hosp)
                
    def suscep_resist(self, liste_edge,liste_resist):
        return super().suscep_resist(liste_edge,liste_resist) 
    
               
    def push_def(self,node_def,hosp):
        list =[]
        for n in range(len(node_def)):
            
            if node_def[n]==1:
    
                if  self.graph.nodes[n]['index'] not in list:
                    list.append(self.graph.nodes[n]['index'])
                    
        super().push_def(list,hosp)
    
    
    def push_defence (self,state ,liste_edge,edge_def,honeypot): 
        li = []
       
        for n in range(len(state)): 
            if state[n]==1:    
                if  self.graph.nodes[n]['index'] not in li:
                    li.append(self.graph.nodes[n]['index'])
                   
        super().push_defence(li,liste_edge,edge_def,honeypot)             
                
    def push_defence_inv (self,state ,liste_edge,hosp): 
        li = []
       
        for n in range(len(state)): 
            if state[n]==1:    
                if  self.graph.nodes[n]['index'] not in li:
                    li.append(self.graph.nodes[n]['index'])
                   
        super().push_defence(li,liste_edge)             
                
                
    def hospot_anime(self,hosp):
        
       
        for n in hosp:
            self.edge.append(n)
        
        super().hospot_anime(hosp)
        
        
    
    def return_compt(self):
        return self.compt
    
    def graphe_ret(self):
        return self.table
   
     
    def step_function(self,sum1):
            
        self.table1 .append(sum1)

    def step_state(self,state):
        listSum = 0
        resist = 0
        susp = 0
        #print(state)
        for n in range(len(state)):
            if (state[n] == 1):
                #print(state[n])
                listSum += state[n]
            elif (state[n] == -1):
                resist = resist + 1
            elif (state[n] == 0):
                susp = susp + 1
                
        self.cumule.append(listSum)
        self.susp.append(susp)
        self.resist.append(resist)
        
        #print('tale de cumule ::::::: ::::', self.cumule)
    def  new_attack(self,liste):
        listSum1 = 0
        for n in range(len(liste)):
            if (liste[n] == 1):
                #print(state[n])
                listSum1 += liste[n]   
        self.new.append(listSum1)        
    def new_return(self) :
        return self.new      
    def step_return(self):
        return self.cumule,self.susp,self.resist
      
    def graphe_ret(self):
        return self.table1
    
    
        

    
    


    
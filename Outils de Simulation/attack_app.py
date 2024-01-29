from calendar import c
import math
import random
from re import U
import matplotlib.pyplot as plt
import numpy


def propagation_unicast(begin,G,q,resist,alpha,prob,nbr,nbr_smart):
        
   for n in range(1):
        l1=[]
        state = [0 for i in range(len(G.nodes))]
        liste_neighbors= []
        # initialement, aucun sommet n'est traité
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        print('nodes',G.nodes)
        
        
        for n in G.nodes():
            liste_neighbors.append(list(G.neighbors(n)))
        
        
        for i in  G.nodes:
                edge=[]
                for j in list(G.neighbors(i)):  
                    edge.append(G[i][j]['index'])
                    
        # on ajoute le sommet de départ aux sommets à traiter    
        state = begin_node(begin,state,G)
        
        q.push(state)
         # on ajoute les node resistant  à traiter   
        liste_resist = resist_list(resist,state,G)
        # on ajoute les honeypot à traiter

        #honeypot1 =honeypot(G,liste_resist,nbr,l1,0)
        honeypot1 = honey_pot(G,l1,l1,resist,l1,liste_resist,nbr,0)

        q.push_resist(state,liste_resist,honeypot1)
        q.hospot_anime(honeypot1)
        
        q.step_state(state)
        #print(state)
        
        
        while ( (0  in state) or  (1  in state) ):
               
                
                    liste = liste = [0] * len(state)
                    liste_SR = [0] * len(state)
                    nv_node_liste = []
                    node_liste = []
                    edges = []
                    liste_edge =[]
                    liste_node  =[]
                    nodes_out=[]
                    nodes_in=[]
                    #de S------R
                   
                    Suscep_Resist(G,state,liste,alpha,q,resist) 
                    liste_resist = resist_list(resist,state,G) 
                    #print(resist)
                    for n in  range(len(state)):  
                        if (state[n]==1): 
                            node = n
                            positions=G.nodes[n]['index']
                            node_liste.append(n)
                            neighbor = liste_neighbors[positions][:]
                            t = random.choice(neighbor)
                            
                            
                            for j in G.nodes:
                                if t == j:
                                    if not G.nodes[j]['visited']:
                                    
                                        G.nodes[j]['visited'] = True
                                        #print(G[node][j]['index'])
                                        edges.append(G[node][j]['index'])
                                        liste[j] = 1
                                        nv_node_liste.append(j)
                                        
                                        if G[node][j]['index'] in honeypot1:
                                            liste_edge.append(G[node][j]['index'])
                                            nodes_out.append(node)
                                            nodes_in.append(j)
                                            liste_node.append(node)
                                            liste_node.append(j)                                            
                            
                                            for j in G.nodes:
                                                for n in liste_node:
                                                    if n == j:
                                                        liste_SR[j] = 1             
                                                      
   
                                
                    edge_def1=edge_def(G,liste_node,liste_resist) 
                    q.push_resist(liste,liste_resist,honeypot1)
                    q.push_edges(edges,honeypot1) #changer le couleur de edge en rouge 
                    q.push(liste) #changer le couleur des node en rouge 
                    q.push_defence(liste_SR,liste_edge,edge_def1,honeypot1)#pour les node defence 
                    q.push_inv_edges(edges,honeypot1)#pour les edge rouge         
                    
                    for i in G.nodes:
                        for n in liste_node:
                            if n == i:
                                liste[i] = 0
                                state[i] = 0                              
                                                                    
                                G.nodes[i]['visited'] = False
                    
                    
                    liste_IS = [0] * len(state)
                    node_IS = []
                    for n in  range(len(state)):
                        if (state[n]==1):
                            state[n]=(numpy.random.choice([0,1], p=[prob,(1-prob)]))
                            if (state[n]== 0):
                                G.nodes[n]['visited'] = False
                                liste_IS[n]==1
                                node_IS.append(n)
                    #print(node_IS)
                    
                            
                    edge_IS = edge_def(G,node_IS,liste_resist)
                    #print('le nodes est ',node_IS , 'les edges est ',edge_IS)
                    q.push_IS(node_IS,edge_IS,honeypot1)

                    for i in range (len(state)):
                        #if liste[i]==1:
                        state[i] = state[i] + liste[i] 
                        

                    q.new_attack(liste)
                    #print('state est ',state)
                    q.step_state(state)

                    q.suscep_resist(honeypot1,liste_resist)
                    #print(edge_def1)
                    #honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist)
                    honeypot1 = honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist,nbr,nbr_smart)
    
                    #honeypot1 = honeypot(G,liste_resist,nbr,edge_def1,nbr_smart)
                    q.hospot_anime(honeypot1)
                    #q.compteur(compt)





def propagation_probability(begin,G,q,resist,alpha,prob,nbr,P,nbr_smart):
        
   for n in range(1):
       
        state = [0 for i in range(len(G.nodes))]
        liste_neighbors= []
       
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        print('nodes',G.nodes)
        
        
        for n in G.nodes():
            liste_neighbors.append(list(G.neighbors(n)))
        
        
        for i in  G.nodes:
                edge=[]
                for j in list(G.neighbors(i)):  
                    edge.append(G[i][j]['index'])
            
                #print('le node i ',i , 'est ',edge  )
        
        state = begin_node(begin,state,G)
        l1=[]
        q.push(state)
        
        liste_resist = resist_list(resist,state,G)
        #print(liste_resist)
        honeypot1 = honey_pot(G,l1,l1,resist,l1,liste_resist,nbr,0)
       
        q.push_resist(state,liste_resist,honeypot1)
        q.hospot_anime(honeypot1)
        q.step_function(1) 
        q.step_state(state)
      
        
        while ( (0  in state) or  (1  in state) ):
            
                
                    liste = liste = [0] * len(state)
                    liste_SR = [0] * len(state)
                    liste_def = [0] * len(state)
                    nv_node_liste = []
                    node_liste = []
                    edges = []
                    liste_edge =[]
                    liste_node  =[]
                    nodes_out=[]
                    nodes_in=[]
                   
                    Suscep_Resist(G,state,liste,alpha,q,resist) 
                    liste_resist = resist_list(resist,state,G) 
                    
                    for n in  range(len(state)):  
                        if (state[n]==1): 
                            node = n
                            positions=G.nodes[n]['index']
                            node_liste.append(n)
                            neighbor = liste_neighbors[positions][:]
                            
                           
                            nb_attack = math.ceil(len(neighbor)*P)

                            #print(node , ':::::::::::::::::::',neigh_attack,'kkkkkkkkkkk',nb,'jjjjjj',liste_ran)
                            #if len(neigh_attack)>= nb_attack:
                                
                            liste_random = random.sample(neighbor, k=nb_attack)
                                                  
                            #else :
                                #liste_random = neigh_attack
                                
                            for j in G.nodes:
                                if not G.nodes[j]['visited']:
                                #if liste_ran is not None:
                                    for i in liste_random:
                                        #if j  in liste_random:
                                        if i == j:
                                            if not G.nodes[j]['visited']:
                                                G.nodes[j]['visited'] = True
                                                edges.append(G[node][j]['index'])
                                                liste[j] = 1
                                                nv_node_liste.append(j)
                                                
                                                if G[node][j]['index'] in honeypot1:
                                                    liste_edge.append(G[node][j]['index'])
                                                    liste_node.append(node)
                                                    #state[n] = 0 
                                                    nodes_out.append(node)
                                                    nodes_in.append(j)
                                                    liste_def[j]== 1
                                                    liste_node.append(j)
                                                    #G.nodes[node]['visited'] = False
                                                    #G.nodes[j]['visited'] = False
                                                    
                                                    for i in G.nodes:
                                                        for n in liste_node:
                                                            if n == i:
                                                                liste_SR[i] = 1 
                                                                
                                                                #liste[i]  = 0       
                                                                #G.nodes[i]['visited'] = False
                                                                     
                            
                            
                            #if not(liste_SR.count(liste_SR[0]) == len(liste_SR)):
                            #    for i in range (len(state)):
                                
                            #        state[i] = state[i] - liste_SR[i]
                    
                        
                    edge_def1=edge_def(G,liste_node,liste_resist) 
                    q.push_resist(liste,liste_resist,honeypot1)
                    q.push_edges(edges,honeypot1) #changer le couleur de edge en rouge 
                    q.push(liste) #changer le couleur des node en rouge 
                    q.push_defence(liste_SR,liste_edge,edge_def1,honeypot1)#pour les node defence 
                    q.push_inv_edges(edges,honeypot1)#pour les edge rouge         
                    
                    for i in G.nodes:
                        for n in liste_node:
                            if n == i:
                                liste[i] = 0
                                state[i] = 0                              
                                                                    
                                G.nodes[i]['visited'] = False
                    
                    
                    liste_IS = [0] * len(state)
                    node_IS = []
                
                    for n in  range(len(state)):
                        if (state[n]==1):
                            state[n]=(numpy.random.choice([0,1], p=[prob,(1-prob)]))
                            if (state[n]== 0):
                                G.nodes[n]['visited'] = False
                                liste_IS[n]==1
                                node_IS.append(n)
                    #print(node_IS)
                    
                    sum1 = sum(liste)        
                    edge_IS = edge_def(G,node_IS,liste_resist)
                    #print('le nodes est ',node_IS , 'les edges est ',edge_IS)
                    q.push_IS(node_IS,edge_IS,honeypot1)

                    for i in range (len(state)):
                        if liste[i]==1:
                            state[i] = state[i] + liste[i] 
                        
                        
                    
                        
                    
                    
                    #print('state est ',state)
                    q.new_attack(liste)
                    q.step_state(state) 
                
                
                    q.suscep_resist(honeypot1,liste_resist)
                    honeypot1 = honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist,nbr,nbr_smart)

                    q.hospot_anime(honeypot1)
 

def propagation_broadcast(begin,G,q,resist,alpha,prob,nbr,nbr_smart):
        
   for n in range(1):
       
        state = [0 for i in range(len(G.nodes))]
        liste_neighbors= []
       
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        print('nodes',G.nodes)
        
        
        for n in G.nodes():
            liste_neighbors.append(list(G.neighbors(n)))
        
        
        for i in  G.nodes:
                edge=[]
                for j in list(G.neighbors(i)):  
                    edge.append(G[i][j]['index'])
            
                #print('le node i ',i , 'est ',edge  )
        
        state = begin_node(begin,state,G)
        l1=[]
        q.push(state)
        
        liste_resist = resist_list(resist,state,G)
        #print(liste_resist)
        honeypot1 = honey_pot(G,l1,l1,resist,l1,liste_resist,nbr,0)
       
        q.push_resist(state,liste_resist,honeypot1)
        q.hospot_anime(honeypot1)
        q.step_function(1) 
        q.step_state(state)
      
        
        while ( (0  in state) or  (1  in state) ):
            
                
                    liste = liste = [0] * len(state)
                    liste_SR = [0] * len(state)
                    liste_def = [0] * len(state)
                    nv_node_liste = []
                    node_liste = []
                    edges = []
                    liste_edge =[]
                    liste_node  =[]
                    nodes_out=[]
                    nodes_in=[]
                    Suscep_Resist(G,state,liste,alpha,q,resist) 
                    liste_resist = resist_list(resist,state,G) 
                    
                    for n in  range(len(state)):  
                        if (state[n]==1): 
                            node = n
                            positions=G.nodes[n]['index']
                            node_liste.append(n)
                            neighbor = liste_neighbors[positions][:]
                            
                            for t in neighbor:
                                for n in G.nodes:
                                    if not G.nodes[n]['visited']:
                                        if t == n:
                                            G.nodes[n]['visited'] = True
                                            edges.append(G[node][t]['index'])
                                            liste[n] = 1
                                            nv_node_liste.append(j)
 
                                            if G[node][t]['index'] in honeypot1:
                                                liste_edge.append(G[node][t]['index'])
                                                liste_node.append(node)
                                                liste_node.append(t)
                                                nodes_out.append(node)
                                                nodes_in.append(t)

                                            for j in G.nodes:
                                                for n in liste_node:
                                                    if n == j:
                                                        liste_SR[j] = 1             
                        
                    
                        
                    edge_def1=edge_def(G,liste_node,liste_resist) 
                    q.push_resist(liste,liste_resist,honeypot1)
                    q.push_edges(edges,honeypot1) #changer le couleur de edge en rouge 
                    q.push(liste) #changer le couleur des node en rouge 
                    q.push_defence(liste_SR,liste_edge,edge_def1,honeypot1)#pour les node defence 
                    q.push_inv_edges(edges,honeypot1)#pour les edge rouge         
                    
                    for i in G.nodes:
                        for n in liste_node:
                            if n == i:
                                liste[i] = 0
                                state[i] = 0                              
                                                                    
                                G.nodes[i]['visited'] = False
                    
                    
                    liste_IS = [0] * len(state)
                    node_IS = []
                
                    for n in  range(len(state)):
                        if (state[n]==1):
                            state[n]=(numpy.random.choice([0,1], p=[prob,(1-prob)]))
                            if (state[n]== 0):
                                G.nodes[n]['visited'] = False
                                liste_IS[n]==1
                                node_IS.append(n)
                    #print(node_IS)
                    
                    sum1 = sum(liste)        
                    edge_IS = edge_def(G,node_IS,liste_resist)
                    #print('le nodes est ',node_IS , 'les edges est ',edge_IS)
                    q.push_IS(node_IS,edge_IS,honeypot1)

                    for i in range (len(state)):
                        if liste[i]==1:
                            state[i] = state[i] + liste[i] 
                        
                        
                    
                        
                    
                    
                    #print('state est ',state)
                    q.new_attack(liste)
                    q.step_state(state) 
                
                
                    q.suscep_resist(honeypot1,liste_resist)
                    honeypot1 = honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist,nbr,nbr_smart)

                    q.hospot_anime(honeypot1)
 
                
def propagation_deterministic_smart(begin,G,q,resist,alpha,prob,nbr,nbr_smart):                
          
   for n in range(1):
       
        state = [0 for i in range(len(G.nodes))]
        liste_neighbors= []
       
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        print('nodes',G.nodes)
        
        
        for n in G.nodes():
            liste_neighbors.append(list(G.neighbors(n)))
        
        
        for i in  G.nodes:
                edge=[]
                for j in list(G.neighbors(i)):  
                    edge.append(G[i][j]['index'])
            
                #print('le node i ',i , 'est ',edge  )
        
        state = begin_node(begin,state,G)
        l1=[]
        q.push(state)
        
        liste_resist = resist_list(resist,state,G)
        #print(liste_resist)
        honeypot1 = honey_pot(G,l1,l1,resist,l1,liste_resist,nbr,0)
       
        q.push_resist(state,liste_resist,honeypot1)
        q.hospot_anime(honeypot1)
        q.step_function(1) 
        q.step_state(state)
      
        
        while ( (0  in state) or  (1  in state) ):
            
                    liste = liste = [0] * len(state)
                    liste_SR = [0] * len(state)
                    liste_def = [0] * len(state)
                    nv_node_liste = []
                    node_liste = []
                    edges = []
                    liste_edge =[]
                    liste_node  =[]
                    nodes_out=[]
                    nodes_in=[]
                    Suscep_Resist(G,state,liste,alpha,q,resist) 
                    liste_resist = resist_list(resist,state,G) 
                    
                   
                    for n in range(len(state)):
                        if (state[n]==1):
                            node = n
                            positions=G.nodes[n]['index']
                            neighbor = liste_neighbors[positions][:]
                            list1=[]
                            table_neigh =[]
                            liste_node_max =[]
                            #print('liste',resist)
                            for j in neighbor:
                                if j  not in resist:
                                    list1.append(j)
                            
                            for i in list1:
                                list_smart= []
                                for j in (list(G.neighbors(i))):
                                    
                                    if j  not in resist:           
                                        list_smart.append(j)
                                table_neigh.append(len(list_smart))
                                maxi = max(table_neigh) 
                            
                            for n in range(len(table_neigh)):
                                
                                if table_neigh[n]==maxi:
                                    liste_node_max.append(list1[n])
                            
                            if (len(liste_node_max)!=0):
                                t=random.choice(liste_node_max)
                            
                                #resist.append(t)
                            for n in G.nodes:
                                
                                    if not G.nodes[n]['visited']:
                                        if t == n:
                                            G.nodes[n]['visited'] = True
                                        
                                            edges.append(G[node][t]['index'])
                                            liste[n] = 1 
                                            
                                            if G[node][t]['index'] in honeypot1:
                                                liste_edge.append(G[node][t]['index'])
                                                liste_node.append(node)
                                                liste_node.append(t)
                                                nodes_out.append(node)
                                                nodes_in.append(t)
                                                for j in G.nodes:
                                                    for n in liste_node:
                                                        if n == j:
                                                            liste_SR[j] = 1             
                        
                    
                        
                    edge_def1=edge_def(G,liste_node,liste_resist) 
                    q.push_resist(liste,liste_resist,honeypot1)
                    q.push_edges(edges,honeypot1) #changer le couleur de edge en rouge 
                    q.push(liste) #changer le couleur des node en rouge 
                    q.push_defence(liste_SR,liste_edge,edge_def1,honeypot1)#pour les node defence 
                    q.push_inv_edges(edges,honeypot1)#pour les edge rouge         
                    
                    for i in G.nodes:
                        for n in liste_node:
                            if n == i:
                                liste[i] = 0
                                state[i] = 0                              
                                                                    
                                G.nodes[i]['visited'] = False
                    
                    
                    liste_IS = [0] * len(state)
                    node_IS = []
                
                    for n in  range(len(state)):
                        if (state[n]==1):
                            state[n]=(numpy.random.choice([0,1], p=[prob,(1-prob)]))
                            if (state[n]== 0):
                                G.nodes[n]['visited'] = False
                                liste_IS[n]==1
                                node_IS.append(n)
                    #print(node_IS)
                    
                    sum1 = sum(liste)        
                    edge_IS = edge_def(G,node_IS,liste_resist)
                    #print('le nodes est ',node_IS , 'les edges est ',edge_IS)
                    q.push_IS(node_IS,edge_IS,honeypot1)

                    for i in range (len(state)):
                        if liste[i]==1:
                            state[i] = state[i] + liste[i] 
                        
                        
                    
                        
                    
                    
                    #print('state est ',state)
                    q.new_attack(liste)
                    q.step_state(state) 
                
                
                    q.suscep_resist(honeypot1,liste_resist)
                    honeypot1 = honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist,nbr,nbr_smart)

                    q.hospot_anime(honeypot1)          

                
def propagation_probabilistic_smart(begin,G,q,resist,alpha,prob,nbr,nbr_smart):
          
   for n in range(1):
       
        state = [0 for i in range(len(G.nodes))]
        liste_neighbors= []
       
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        print('nodes',G.nodes)
        
        
        for n in G.nodes():
            liste_neighbors.append(list(G.neighbors(n)))
        
        
        for i in  G.nodes:
                edge=[]
                for j in list(G.neighbors(i)):  
                    edge.append(G[i][j]['index'])
            
                #print('le node i ',i , 'est ',edge  )
        
        state = begin_node(begin,state,G)
        l1=[]
        q.push(state)
        
        liste_resist = resist_list(resist,state,G)
        #print(liste_resist)
        honeypot1 = honey_pot(G,l1,l1,resist,l1,liste_resist,nbr,0)
       
        q.push_resist(state,liste_resist,honeypot1)
        q.hospot_anime(honeypot1)
        q.step_function(1) 
        q.step_state(state)
      
        
        while ( (0  in state) or  (1  in state) ):
            
                    liste = liste = [0] * len(state)
                    liste_SR = [0] * len(state)
                    liste_def = [0] * len(state)
                    nv_node_liste = []
                    node_liste = []
                    edges = []
                    liste_edge =[]
                    liste_node  =[]
                    
                    nodes_out=[]
                    nodes_in=[]
                    
                    Suscep_Resist(G,state,liste,alpha,q,resist) 
                    liste_resist = resist_list(resist,state,G) 
                    
                   
                    for n in range(len(state)):
                        if (state[n]==1):
                            node = n
                            positions=G.nodes[n]['index']
                            neighbor = liste_neighbors[positions][:]
                            #nb_attack = math.ceil(len(neighbor)*P)

                            list1=[]
                            table_neigh =[]
                            liste_node_max =[]
                            neigh_not_max = []
                            table_neigh_max =[]
                            list_not_max  =[]
                            liste2=[]
                            #print('liste',resist)
                            for j in neighbor:
                                if j  not in resist:
                                    list1.append(j)
                            
                            for i in list1:
                                list_smart= []
                                for j in (list(G.neighbors(i))):
                                    
                                    if j  not in resist:           
                                        list_smart.append(j)
                                table_neigh.append(len(list_smart))
                            
                                maxi = max(table_neigh) 
                            
                            for n in range(len(table_neigh)):
                                
                                if table_neigh[n]==maxi:
                                    liste_node_max.append(list1[n])
                                else:
                                    list_not_max.append(list1[n])
                            
                            for n  in list_not_max:
                                for j in (list(G.neighbors(n))):
                                    if j  not in resist:           
                                        neigh_not_max.append(j)
                                
                                table_neigh_max.append(len(neigh_not_max))
                                    
                            #print(maxi)    
                            nbr_not=random.randint(0, len(list_not_max))
                            
                            if nbr_not != 0 :
                                
                                liste2=random.choices(list_not_max,weights=table_neigh_max,k=nbr_not)  
                                
                                liste_node_max.extend(liste2)
                               
                            #print(liste_node_max)
                            
                        
                            
                            for t in liste_node_max:
                                for n in G.nodes:
                                    #print('letat de state de node',n,'est :::::',state[n])
                                    if not G.nodes[n]['visited']:
                                        if t == n:
                                            G.nodes[n]['visited'] = True
                                        
                                            edges.append(G[node][t]['index'])
                                            liste[n] = 1  
                                                                             
                                            if G[node][t]['index'] in honeypot1:
                                                    liste_edge.append(G[node][t]['index'])
                                                    liste_node.append(node)
                                                    liste_node.append(t)
                                                    nodes_out.append(node)
                                                    nodes_in.append(t)

                                                    for j in G.nodes:
                                                        for n in liste_node:
                                                            if n == j:
                                                                liste_SR[j] = 1        
                        
                    
                        
                    edge_def1=edge_def(G,liste_node,liste_resist) 
                    q.push_resist(liste,liste_resist,honeypot1)
                    q.push_edges(edges,honeypot1) #changer le couleur de edge en rouge 
                    q.push(liste) #changer le couleur des node en rouge 
                    q.push_defence(liste_SR,liste_edge,edge_def1,honeypot1)#pour les node defence 
                    q.push_inv_edges(edges,honeypot1)#pour les edge rouge         
                    
                    for i in G.nodes:
                        for n in liste_node:
                            if n == i:
                                liste[i] = 0
                                state[i] = 0                              
                                                                    
                                G.nodes[i]['visited'] = False
                    
                    
                    liste_IS = [0] * len(state)
                    node_IS = []
                
                    for n in  range(len(state)):
                        if (state[n]==1):
                            state[n]=(numpy.random.choice([0,1], p=[prob,(1-prob)]))
                            if (state[n]== 0):
                                G.nodes[n]['visited'] = False
                                liste_IS[n]==1
                                node_IS.append(n)
                    #print(node_IS)
                    
                    sum1 = sum(liste)        
                    edge_IS = edge_def(G,node_IS,liste_resist)
                    #print('le nodes est ',node_IS , 'les edges est ',edge_IS)
                    q.push_IS(node_IS,edge_IS,honeypot1)

                    for i in range (len(state)):
                        if liste[i]==1:
                            state[i] = state[i] + liste[i] 
                        
                        
                    
                        
                    
                    
                    #print('state est ',state)
                    q.new_attack(liste)
                    q.step_state(state) 
                
                
                    q.suscep_resist(honeypot1,liste_resist)
                    honeypot1 = honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist,nbr,nbr_smart)

                    q.hospot_anime(honeypot1)          
            
           
def propagation_broadcast_smart(begin,G,q,resist,alpha,prob,nbr,nbr_smart):
           
   for n in range(1):
       
        state = [0 for i in range(len(G.nodes))]
        liste_neighbors= []
       
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        print('nodes',G.nodes)
        
        
        for n in G.nodes():
            liste_neighbors.append(list(G.neighbors(n)))
        
        
        for i in  G.nodes:
                edge=[]
                for j in list(G.neighbors(i)):  
                    edge.append(G[i][j]['index'])
            
                #print('le node i ',i , 'est ',edge  )
        
        state = begin_node(begin,state,G)
        l1=[]
        q.push(state)
        
        liste_resist = resist_list(resist,state,G)
        #print(liste_resist)
        honeypot1 = honey_pot(G,l1,l1,resist,l1,liste_resist,nbr,0)
       
        q.push_resist(state,liste_resist,honeypot1)
        q.hospot_anime(honeypot1)

        q.step_state(state)
      
        
        while ( (0  in state) or  (1  in state) ):
            
                for k in range(2):
                    liste = liste = [0] * len(state)
                    liste_SR = [0] * len(state)
                    liste_def = [0] * len(state)
                    edges = []
                    liste_edge =[]
                    liste_node  =[]
                    nodes_out = []
                    nodes_in = []
                   
                    Suscep_Resist(G,state,liste,alpha,q,resist) 
                    liste_resist = resist_list(resist,state,G) 
                    
                   
                    for n in range(len(state)):
                        if (state[n]==1):
                            node = n
                            positions=G.nodes[n]['index']
                            neighbor = liste_neighbors[positions][:]
                            list1=[]
                        
                            table_neigh =[]
                            liste_node_max =[]
                            
                            for j in neighbor:
                                if j  not in resist:
                                    list1.append(j)
                            
                            for i in list1:
                                list_smart= []
                                for j in (list(G.neighbors(i))):
                                    if j  not in resist:
                                        list_smart.append(j)
                                
                                table_neigh.append(len(list_smart))
                                maxi = max(table_neigh)
                            
                            for n in range(len(table_neigh)):
                                
                                if table_neigh[n]==maxi:
                                    liste_node_max.append(list1[n])
                            
                            #resist.extend(liste_node_max)
                        
                            
                            for t in liste_node_max:
                                for n in G.nodes:
                                    #print('letat de state de node',n,'est :::::',state[n])
                                    if not G.nodes[n]['visited']:
                                        if t == n:
                                            G.nodes[n]['visited'] = True
                                        
                                            edges.append(G[node][t]['index'])
                                            liste[n] = 1  
                                                                             
                                            if G[node][t]['index'] in honeypot1:
                                                    liste_edge.append(G[node][t]['index'])
                                                    liste_node.append(node)
                                                    liste_node.append(t)
                                                    nodes_out.append(node)
                                                    nodes_in.append(t)

                                                    for j in G.nodes:
                                                        for n in liste_node:
                                                            if n == j:
                                                                liste_SR[j] = 1             
                        
                                     
                        
                    
                        
                    edge_def1=edge_def(G,liste_node,liste_resist) 
                    q.push_resist(liste,liste_resist,honeypot1)
                    q.push_edges(edges,honeypot1) #changer le couleur de edge en rouge 
                    q.push(liste) #changer le couleur des node en rouge 
                    q.push_defence(liste_SR,liste_edge,edge_def1,honeypot1)#pour les node defence 
                    q.push_inv_edges(edges,honeypot1)#pour les edge rouge         
                    
                    for i in G.nodes:
                        for n in liste_node:
                            if n == i:
                                liste[i] = 0
                                state[i] = 0                              
                                                                    
                                G.nodes[i]['visited'] = False
                    
                    
                    liste_IS = [0] * len(state)
                    node_IS = []
                
                    for n in  range(len(state)):
                        if (state[n]==1):
                            state[n]=(numpy.random.choice([0,1], p=[prob,(1-prob)]))
                            if (state[n]== 0):
                                G.nodes[n]['visited'] = False
                                liste_IS[n]==1
                                node_IS.append(n)
                    #print(node_IS)
                    
                    sum1 = sum(liste)        
                    edge_IS = edge_def(G,node_IS,liste_resist)
                    #print('le nodes est ',node_IS , 'les edges est ',edge_IS)
                    q.push_IS(node_IS,edge_IS,honeypot1)

                    for i in range (len(state)):
                        if liste[i]==1:
                            state[i] = state[i] + liste[i] 
                    
                    #print('state est ',state)
                    q.new_attack(liste)
                    q.step_state(state) 
                
                
                q.suscep_resist(honeypot1,liste_resist)
                honeypot1 = honey_pot(G,nodes_out,nodes_in,resist,liste_edge,liste_resist,nbr,nbr_smart)

                q.hospot_anime(honeypot1)          

def edge_def(G,liste_node,liste_resist):
    edge_def =[]
    
    for i in  liste_node:
        for j in list(G.neighbors(i)):
            if (G[i][j]['index'] not in liste_resist):
                if (G[i][j]['index'] not in edge_def):
                    edge_def.append(G[i][j]['index'])   
    return  edge_def


def  Suscep_Resist(G,state,liste,alpha,q,resist):
    Nv_resist = []
    liste_resist2= []
    for n in  range(len(state)):
        
        if (state[n]==0):
            liste[n]=(numpy.random.choice([-1,0], p=[alpha,(1-alpha)]))
            if (liste[n]== -1):
                G.nodes[n]['visited'] = True 
                Nv_resist.append(n)  
    resist.extend(Nv_resist) 

                  
                  
def begin_node(begin,state,G):
    for i in range(len(begin)):  
            for n in range(len(G.nodes)):
                if (n == begin[i]):
                    state[n]= 1
                    G.nodes[n]['visited'] = True
    return state    
        
        
def resist_list(resist,state,G):
    list_neigh = []
    list_edge =[]
    for j in range(len(resist)):
            for n in range(len(G.nodes)):
                
                if(n == resist[j]):
                    state[n]= -1
                    G.nodes[n]['visited'] = True
                    list_neigh.append(list(G.neighbors(resist[j])))
                    
                    for i in list_neigh[j][:]:
                        
                        list_edge.append(G[resist[j]][i]['index'])
    return list_edge

def honey_pot(G,nodes_out,nodes_in,resist,list_edge,liste_resist,nbr_honey,nbr_honey_smart):
    edge_total =[]
    len_smart = []
    neigh = []
    edge_smart=[]
    edge_honey_smart=[]
    
    for n in nodes_out :
        liste_remove=(list(G.neighbors(n)))
        for i in resist:
            if i in liste_remove:
                liste_remove.remove(i)
                
        for k in nodes_in:
            if k in liste_remove:
                liste_remove.remove(k)        
        
        neigh.append(liste_remove)
        
    for k  in range(len(neigh)):
        for i in range(len(neigh[k][:])):
            list_neigh=(list(G.neighbors(neigh[k][i])))
            for i in resist:
                if i in list_neigh:
                    list_neigh.remove(i)
            len_smart.append(len(list_neigh))  
              
    for k in  range(len(neigh)):
        for i in range(len(neigh[k][:])):
            #edge_smart = G[neigh[k][i]][nodes_out[k]]['index'] 
            
            edge_smart.append(G[neigh[k][i]][nodes_out[k]]['index'] )    

    for n in G.nodes():
            list_ne=(list(G.neighbors(n)))            
            for i in range(len(list_ne)):
                edge = G[list_ne[i]][n]['index']
                if edge not in list_edge:
                    if edge not in edge_smart:
                        if edge not in liste_resist:
                            if edge not in edge_total :
                    
                                edge_total.append(edge)  
    taille = len(edge_total)
    tail_smart =len(edge_smart)
    x=nbr_honey-nbr_honey_smart
    
    if tail_smart == 0 :
        
        if taille >=nbr_honey:
            
            hosp=random.sample(edge_total, k=nbr_honey)
         
        else:
            hosp =random.sample(edge_total, k=taille)
    else :
        if taille >=nbr_honey:
            #hosp=random.sample(edge_total, k=x) 
            if tail_smart >= nbr_honey_smart:
                honey_smart= random.choices(edge_smart,weights=len_smart,k=nbr_honey_smart)                

                hosp=random.sample(edge_total, k=x) 
                
            else:
                y=(nbr_honey-tail_smart)
                honey_smart= random.choices(edge_smart,weights=len_smart,k=tail_smart)

                
                hosp=random.sample(edge_total, k=y)
                
        else :
            hosp =random.sample(edge_total, k=taille)
            z=nbr_honey-taille
            if tail_smart >= z:
                
                honey_smart= random.choices(edge_smart,weights=len_smart,k=z)

            else:
                if tail_smart >= nbr_honey_smart:
                    honey_smart= random.choices(edge_smart,weights=len_smart,k=nbr_honey_smart)
  
                else:
                    honey_smart= random.choices(edge_smart,weights=len_smart,k=tail_smart)

        
        hosp.extend(honey_smart)
        #print('hosp est ',hosp , 'le hosp smart ', honey_smart)
    #print('les hosp est ',hosp)           
    return hosp         
    
   

            

def plot_cumulative(q):
      
    liste = []
    liste1 = []
    liste2= []

    
    liste,liste1,liste2 = q.step_return()
    #print('step des cumulative ' ,  liste )
    #x = list(range(1,len(liste)+1))
    #print(x)
    return liste,liste1,liste2

          
def plot_NV(q) :
    
    y = []
   
    y = q.new_return()
  
    
    return y
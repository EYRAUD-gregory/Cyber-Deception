
from timeit import repeat
from numpy import block, size
from matplotlib.animation import writers
import attack_app
import animation_attack
import graph_type
import networkx as nx
from IPython.display import HTML
import random
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import matplotlib.pyplot as plt
import time
from matplotlib import gridspec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from itertools import count
from IPython import display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt #plotting
from datetime import datetime, timedelta #data creation 
from celluloid import Camera #animation
plt.style.use('seaborn')

#enregistrer video 

prompt =""" 
       
                
  """
  
  
prompt = """  """
prompt = """ 
        vous voulez enregistrer votre animation sous forme une vidéo:
        oui :
        non :
                
  """
choicevideo = input(prompt)
  
   
prompt = """  """
prompt = """ 
        Vous voulez  creer un nouveau graphe  : 
        oui :
        non :
                
  """
choice = input(prompt)


if choice == 'oui' :
    size = int(input('Entrer le nombre de nœuds : '))
  
    prompt = """Définir le mode génération ou le type du graphe. 
  Génération aléatoire                   :   1  
  Génération par la méthode d'Erdös-Réyni:   2      
  Small Word Network                     :   3  

  """
    choice1 = input(prompt)
    
    if choice1 == '1':
        pts = graph_type.gen_in_disk(size)
        G = graph_type.delaunay_graph(pts)
 	# saving graph created above in gexf format
        nx.write_gexf(G, "geeksforgeeks.gexf")
        
    if choice1 == '2':
        P = float(input('Entrer la probabilité d' 'activation de chaque arête : '))
        
        G = graph_type.erdos_reyni(size, P)
        
        while (not (nx.is_connected(G))):
            prompt = "Cette valeur ne permet pas de générer un graphe connexe."
            P = float(input('Entrer une probabilité plus grande : '))
            G = graph_type.erdos_reyni(size, P)
        nx.write_gexf(G, "geeksforgeeks.gexf")    
        
    if choice1 == '3':
        m = int(input('Entrer le degré de chaque nœud du premier graphe : '))
        P = float(input('Entrer la probabilité de modifier chaque arête   : '))
        G = graph_type.small_word_network(size, m, P)
        nx.write_gexf(G, "geeksforgeeks.gexf")

      
if choice == 'non':
    G = nx.read_gexf( "geeksforgeeks.gexf", node_type=int)
    
    
    
positions = nx.spring_layout(G)
fig, ax = plt.subplots()

nx.draw(G, pos=positions, ax=ax, node_size=150, with_labels=True)

plt.show(block=False)




#return  liste des  node infecté
def node_begin(G):
    begin= []
    n = int(input("Entrer le nombre de nœuds infectés au début du jeu  :"))
    for i in range(0, n):
        j=i+1
        node = int(input ("Entrer l'element , %s :" % j  ))
        begin.append(node)
    return begin

#return liste des  node resistant 

def node_resistance(G,begin):
	resist = []
	n = int(input("Entrer le nombre de nœuds resistants au début du jeu :"))
	for i in range(0, n):
		j=i+1
		node = int(input ("Entrer l'element , %s :" % j  ))
		for k in begin:
			if node == k:
				print('le node est déja choisie commme infecteé')
				node = int(input ("Entrer l'element , %s :" % j  ))


		resist.append(node)
	return resist
#demande le nombre total des honeypot 
def edge_hospot(G):
   
    h = int(input("entrer le nombre des honeypot :"))
    return h

#def smart_hospot(G):
       

#demande le nombre  des smart honeypot
def smart_hospot(G,nbr):
    nb = int(input("entrer le nombre des smart  honeypot :"))
    if (nbr < nb):
        print('entrer plus grand :')
        nb = int(input("entrer le nombre des smart  hospot :"))
    return nb 

        

alpha =float(input('Entrer la probabilité de S vers R: '))
begin = []

Pors =float(input('Entrer la probabilité de I vers S: '))


resist = []
begin = node_begin(G) 
resist = node_resistance(G,begin)
nbr = edge_hospot(G)

q = animation_attack.queue(G)

nbr_smart = smart_hospot(G,nbr)



prompt = """Choisir le rythme de propagation 
  Propagation unicast   : 1  - un pour un 
  Propagation multicast : 2  - proportion à définir
  Propagation broadcast : 3  - un pour tous
  Propagation propagation_deterministic_smart : 4  - un pour un (plus grand degré )
  Propagation propagation_probabilistic_smart :  5 - proportion définir 
  Propagation multicast_smart :  6 - un pour tous les plus grand degré 

  

  """


choice = input(prompt)

if choice == '1':
    attack_app.propagation_unicast(begin, G, q, resist,alpha,Pors,nbr,nbr_smart)

if choice == '2':
    P1 = float(input('Entrer la proportion des voisins à infecter : '))
    attack_app.propagation_probability(begin,G,q,resist,alpha,Pors,nbr,P1,nbr_smart)

if choice == '3':
  attack_app.propagation_broadcast(begin,G,q,resist,alpha,Pors,nbr,nbr_smart)
    
if choice == '4':
	attack_app.propagation_deterministic_smart(begin,G,q,resist,alpha,Pors,nbr,nbr_smart)
if choice == '5':
	P1 = float(input('Entrer la proportion des voisins à infecter : '))
	attack_app.propagation_probabilistic_smart(begin,G,q,resist,alpha,Pors,nbr,nbr_smart)

if choice == '6':
	attack_app.propagation_broadcast_smart(begin,G,q,resist,alpha,Pors,nbr,nbr_smart)
 
    
positions = nx.spring_layout(G)  # positions for all nodes

fig = plt.figure(figsize=(10, 10))

gs = gridspec.GridSpec(2, 2, height_ratios=[3,2])

ax = fig.add_subplot(gs[:-1, :])
ax.clear()

compt=0


def update(frame):
	ax.clear()
	#time_text.set_text("Points: %.0f" % int(num))
	nx.draw(
	    G,
	    pos=positions,
	    ax=ax,
	    node_size=250,
	    with_labels=True,
	    font_size=10,
	    node_color=q.animation['colors'][frame],
	    edge_color=q.animation['colors_edge'][frame],
	    width=q.animation['widths'][frame])
	
ani = matplotlib.animation.FuncAnimation(fig,
                                         update,
                                         frames=len(q.animation['colors']),
                                         interval=1000,
                                         repeat=False)


#plt.show(block=False)

ax2 = fig.add_subplot(gs[-1, :])

l1,l2,l3 = attack_app.plot_cumulative(q)
plt.suptitle( 'Model  Attack-Defense ')

x1, y1 ,y2, y3 = [], [], [], []
plt.grid( which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid( which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Time (step)')
plt.ylabel('Population (Number)')
#plt.ylabel('Population (Number)')


for n in  range(len(l1)) :
    
	x1.append(n)
 
y1=l1

ax2.plot(x1, l1, color="red", label='infected')#infécté
ax2.plot(x1, l2, color="blue", label='susceptible')#suscp
ax2.plot(x1, l3, color="black",label='resistant')#resist


plt.legend()
#

#anim = FuncAnimation(fig, animate, interval=10)
plt.tight_layout()
#plt.show(block=False)
if choicevideo ==  'oui' :
#Uniquement si vous avez installer ffmpeg
    Writer = writers['ffmpeg']
    writer = Writer(
        fps=2,
        metadata=dict(artist="oumaima diami", title = "AnimationPlt"),
        bitrate=8000)

    ani.save("AnimationPlt.mp4", writer=writer)
    HTML(ani.to_html5_video())

    plt.show()
else :

    plt.show()
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np 
import epidemics_helper
import pandas as pd


"""
Given a networkx graph G with nodes attributes coordinates, the function plots the network with every node labelled as susceptible.

G = networkx graph G with nodes attributes coordinates
nodes_status =  numpy array of length time_of_simulation and contains for each index (day) the number of population in each state
title = string which is the title of the plot
ax = matplolib axis to plot on
edge_width = float which indicates the width of the edges to draw
"""
def draw_graph(G,nodes_status=[],title="View of the network",ax=None,edge_width=1.0):
    if not nodes_status:
        nodes_status = np.zeros(len(G.nodes)).tolist()
    G_plagued = G.copy()
    nodes_S = []
    nodes_I = []
    nodes_R = []
    for i in range(len(nodes_status)):
        status = nodes_status[i]
        G_plagued.nodes[i]["state"] = status
        if status == 0:
            nodes_S.append(i)
        elif status == 1:
            nodes_I.append(i)
        else:
            nodes_R.append(i)
    if not ax:
        fig, ax = plt.subplots(1, figsize=(200,120))
        ax.set_title(title, fontsize=150)
    nx.draw_networkx_nodes(G_plagued, pos=nx.get_node_attributes(G,'coordinates'), nodelist=nodes_S, ax=ax, node_color='b', label="Susceptible")
    nx.draw_networkx_nodes(G_plagued, pos=nx.get_node_attributes(G,'coordinates'), nodelist=nodes_I, ax=ax, node_color='r', label="Infected")
    nx.draw_networkx_nodes(G_plagued, pos=nx.get_node_attributes(G,'coordinates'), nodelist=nodes_R, ax=ax, node_color='g', label="Recovered")
    nx.draw_networkx_edges(G_plagued, pos=nx.get_node_attributes(G,'coordinates'), ax=ax, width=edge_width)
    ax.legend(fontsize=150)

"""
Given a simulation from the helper package epidemics, returns a tuple where the first element is a dictionnary with keys = times in the 
the input list time_stamps and values = states of the population, and the second is a numpy array of the repartition of the population in 
function of time with index = days of the simulation.  

SIR = simulation runned using the helper package epidemics
time = max time of the simulation
time_stamps = list of times where we want the exact repartition
"""
def nodes_status_over_time(SIR, time, time_stamps):
    status_count = []
    time_stamps_status_list = {}
    for t in range(time): # we iterate over all the time of the simulation
        status_list = []
        for node in range(SIR.n_nodes): # for each node in the network
            status_list.append(SIR.get_node_status(node,t)) # we append its state in the temp list
        count = np.bincount(np.asarray(status_list)) # using numpy we compute the number of people in each state
        status_count.append(count if count.shape[0] == SIR.STATE_SPACE else np.append(count, np.zeros(len(SIR.STATE_SPACE) - count.shape[0], dtype=int)).tolist()) # the condition is here to handle the case where all the states are not represented in the bincount
        for t_s in time_stamps: # this is where we the information at the given time stamps
            if t == t_s:
                time_stamps_status_list[t] = status_list.copy()
    return time_stamps_status_list, np.asarray(status_count)    

"""
Given the population status at any time of the simulation, plot it's evolution. I.E. the number or percentage of the population
at each state in function of time.

nodes_status: numpy array of length time_of_simulation and contains for each index (day) the number of population in each state
ax = matplolib axis to plot on
percentage = True if you want the percentage rather than the number of people
title = string which is the title of the plot
"""
def plot_population_status(nodes_status, ax=None, percentage=False,title="Evolutions of cases over time"):
    status = nodes_status
    if not ax:
        fig, ax = plt.subplots()
        ax.set_title(title) 
    ax.set_xlabel("time (day)")
    ax.set_ylabel("Number of people")
    if percentage:
        status = status/status[0].sum()
        ax.set_ylabel("% of the population")
    ax.plot(status[:,[0]], label="Susceptible")
    ax.plot(status[:,[1]], label="Infected")
    ax.plot(status[:,[2]], label="Recovered")
    ax.legend()

"""
Given a marker which is a percentage of the population, print the days where this percentage is reached for the different
population states. If day = -1, then it means that the state doesn't reach the marker in the simulation.

nodes_status: numpy array of length time_of_simulation and contains for each index (day) the number of population in each state
marker: percentage of the population we want to investigate for each state of the population
"""
def epidemic_markers(nodes_status, marker):
    status = nodes_status/nodes_status[0].sum()
    susceptible = np.where(status[:,[0]] <= marker)[0][0] if np.where(status[:,[0]] <= marker)[0].size > 0 else -1
    infected = np.where(status[:,[1]] >= marker)[0][0] if np.where(status[:,[1]] >= marker)[0].size > 0  else -1
    recovered = np.where(status[:,[2]] >= marker)[0][0] if np.where(status[:,[2]] >= marker)[0].size > 0  else -1
    print(f"The epidemic has the following evolution : {marker*100}% of the population stop being susceptible at time {susceptible}, ", end="")
    print(f"is infected at time {infected} and has recover at time {recovered}")


"""
Function that extracts the edge betweenness from a csv for all edges. We have previously computed it using the function 
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html#networkx.algorithms.centrality.edge_betweenness_centrality
of the package networkx. As it is really computationnal costly we saved the data in a pandas DataFrame and then as a csv file which was more convenient to work with.

path = string that indicates the path to the csv file
"""
def extract_max_centered_edges(path="../data/betweenness_edges.csv"):
    df_max_centered_edges = pd.read_csv("../data/max_centered_edges.csv", index_col="Unnamed: 0")
    return [tuple(map(lambda x : int(x),x[1:-1].split(","))) for x in df_max_centered_edges["Edge"]] # this function is directly correlated with how the dataframe is structured
    


"""
Functions that simulate the epidemy on a network created from a modified version of the initial network. Then it plots several metrics such
as the network itself and the evolution of the population status.

nodes = list of nodes in the network
edges = list of edges of the network
time = int which is the max run time of the simulation
beta = float which is the beta parameter of the simulation (exp distribution for infection)
gamma = float which is the gamma parameter of the simulation (exp distribution for recovery)
budget = int indicating the max number of edges we can cut
day = int indicating the time stamps at which we should print the metrics
sim_nb = int indicating the max number of simulations 
draw = boolean indicating if the network graph should be draw or not
strategy = string indicating which strategy we should apply to eliminate edges (random, vaccination, betweenness or community)
"""
def strategy_simulation(nodes, edges, time=100, beta=10.0, gamma=0.1, budget=1000, sim_nb=3, day=30, draw=False, strategy="random"):
    assert(time > 0 and beta > 0 and gamma > 0 and budget > 0 and sim_nb > 0 and day > 0 and day <= time and (strategy=="random" or strategy=="vaccination" or strategy=="betweenness" or strategy=="community"))
    fig1, axs1 = plt.subplots(sim_nb, figsize=(10,2*sim_nb)) # create the figure to plot the population status evolution
    fig1.suptitle('Evolution of cases over time')
    axs2 = None
    if draw: # draw the networkx graph if precised
        fig2, axs2 = plt.subplots(sim_nb, figsize=(400,200*sim_nb))
        fig2.suptitle('Epidemic state of the network', fontsize=300)
    if sim_nb == 1: # if there is a single simulation we have only an axis and not a list of axis thus for implementation reason we wrap it into a list
        axs1 = [axs1]
        if draw:
            axs2 = [axs2]
    day_status = np.zeros(3) # initiate the population status at the given day as a numpy array of dim 3 filled with 0s
    purified_edges = edges.copy()
    if strategy == "vaccination": # if the strategy is vaccination we cut edges linking high degree nodes
        purified_edges = cut_edges_max_degree(edges, nodes, budget)
    elif strategy == "betweenness": # if the strategy is betweenness we cut edges with high betweenness
        edges_btw = extract_max_centered_edges()
        for j in range(budget):
            purified_edges.remove(edges_btw[j])
    elif strategy == "community": # if the strategy is community, we locate and isolate the communities in our network
        must_cut_edges = retrieve_community_boundary_edges(nodes, edges)
        if budget < len(must_cut_edges):
            must_cut_edges = must_cut_edges[:budget]
        purified_edges = list(set(edges) - set(must_cut_edges))
    for i in range(sim_nb):
        if strategy == "random": # if the strategy is random, we shuffle the list of edges and keep only the required number
            purified_edges = edges.copy() # we put the condition in the loop because we want the simulation to be each time different to have a better "idea" due to randomness
            np.random.shuffle(purified_edges)
            purified_edges = purified_edges[:len(edges)-budget]
        sim_and_draw(nodes, purified_edges, time, beta, gamma, day, day_status, draw, i, sim_nb, axs1, axs2=axs2) # simulate the epidemy and draw if asked the network
    day_status = day_status / sim_nb
    print(f"By following strategy {strategy}, on day {day}, the average of susceptible people is {day_status[0]}, of infected is {day_status[1]} and of recovered is {day_status[2]}")


"""
Given a list of nodes and edges which compose a graph, compute the communities of the graph using the Clauset-Newman-Moore greedy modularity maximization
(https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)
of the package networkx and then returns the boudary edges between these communities.

nodes = list of nodes in the network
edges = list of edges of the network
"""
def retrieve_community_boundary_edges(nodes, edges):
    G_init = nx.Graph()
    G_init.add_nodes_from(nodes)
    G_init.add_edges_from(edges)
    communities = nx.algorithms.community.greedy_modularity_communities(G_init)
    list_must_cut_edges = []
    for i in range(len(communities)): # we iterate between each non equal communities and compute their boundary edges
        for j in range(len(communities)):# i.e. edges that have a node in community "a" and a node in community "b"
            if j != i:
                list_must_cut_edges += [e for e in nx.edge_boundary(G_init, communities[i], nbunch2=communities[j])]
    list_must_cut_edges = list(set(list_must_cut_edges) - (set(list_must_cut_edges) -set(edges))) # we do this to get rid of duplicated edges
    return list_must_cut_edges                                                                   # as a boundary edge will appear two times : (a,b) and (b,a)

"""
Given a networkx graph G with node attribute coordinates and its communities, it will plot the different communities of this graph.

G = networkx graph G with nodes attributes coordinates
communities = list of frozen-set of nodes which represents the communities
i = index of the community to plot, if -1 it will plot them all
boundaries = True if the boundary edges must be plotted
"""
def draw_communities(G,communities,i=-1, boundaries=False):
    fig, ax = plt.subplots(1, figsize=(200,120))
    if i != -1:
        nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G,'coordinates'), nodelist=list(communities[i]), ax=ax, node_color="b")
        nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G,'coordinates'), nodelist=list(set(list(G.nodes)) - set(list(communities[i]))), ax=ax, node_color="r")
    else:
        colors = plt.cm.get_cmap('hsv', len(communities))
        for j in range(len(communities)):
            nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G,'coordinates'), nodelist=list(communities[j]), ax=ax, node_color=colors(j))
        if boundaries:
            boundary_edges = retrieve_community_boundary_edges(list(G.nodes), list(G.edges))
            nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G,'coordinates'), edgelist=boundary_edges, ax=ax, width=30)

"""
Given a list of edges, a list of nodes and a budget, it returns a edge_list where the edges which were linked to high degree
nodes have been cutted (a budget number of them).

nodes = list of nodes in the network
edges = list of edges of the network
budget = int representing the number of edges to cut
"""
def cut_edges_max_degree(edges, nodes, budget):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    degrees_sorted = sorted(list(G.degree()), key = lambda t : t[1], reverse=True) # sort the nodes by their degree
    new_edges_list = edges.copy()
    cost = 0
    for node in degrees_sorted: # for each node in descending degree order
        node_to_kill = node[0]
        for edge in new_edges_list: # for each edges
            if(cost >= budget): # if we attained the cost we return
                return new_edges_list
            if edge[0] == node_to_kill or edge[1] == node_to_kill : # else we check if the edge is linked to the node in question and if yes we cut it
                new_edges_list.remove(edge)
                cost+=1
    return new_edges_list


"""
Functions that simulate the epidemy on a network created from a modified version of the initial network. Then it plots several metrics such
as the network itself and the evolution of the population status.

nodes = list of nodes in the network
edges = list of edges of the network
time = int which is the max run time of the simulation
beta = float which is the beta parameter of the simulation (exp distribution for infection)
gamma = float which is the gamma parameter of the simulation (exp distribution for recovery)
day = int indicating the time stamps at which we should print the metrics 
day_status = list of the population status at the day precised in input
draw = boolean indicating if the network graph should be draw or not
sim_id = int representing the id, i.e. the index of the actual run
sim_nb = int indicating the max number of simulations
axs1 = matplotlib axis to plot the state evolution 
axs2 = matplotlib axis to plot the networkx
"""
def sim_and_draw(nodes, purified_edges, time, beta, gamma, day, day_status, draw, sim_id, sim_nb, axs1, axs2=None):
        # we create a graph
        G_sim = nx.Graph()
        G_sim.add_nodes_from(nodes)
        G_sim.add_edges_from(purified_edges)
        # we simulate an epidemy
        SIR = epidemics_helper.SimulationSIR(G_sim, beta, gamma)
        SIR.launch_epidemic(source=np.random.randint(0,len(nodes)), max_time=time)
        status_list, nodes_status_count = nodes_status_over_time(SIR, time, [day]) # we retrive the population status over time
        if sim_nb != 1: # give a subtitle if multiple simulations
            axs1[sim_id].set_title("Simulation "+str(sim_id+1))
        plot_population_status(nodes_status_count, ax=axs1[sim_id], percentage=True)
        if draw: # draw the graph is indicated
            if sim_nb != 1: # give a subtitle if multiple simulations
                axs2[sim_id].set_title("Simulation "+str(sim_id+1), fontsize=300)
            draw_graph(G_sim, nodes_status=status_list[day], ax=axs2[sim_id])
        day_status += nodes_status_count[day] # contribute to the population status at the given day in order to compute the average


"""
Functions that simulate the epidemy on a network created from a modified version of the initial network. Then it returns the number of susceptible at the given
day. It is used to compute the average of susceptible for a given strategy and budget. It is really similar to simulate_strategy but here there is no question
of plotting, thus we do not polute the output section with a lot of plots. 

nodes = list of nodes in the network
edges = list of edges of the network
budget = int indicating the max number of edges we can cut
strategy = string indicating the type of simulation we have to run (random, vaccination, betweenness or community)
time = int which is the max run time of the simulation
beta = float which is the beta parameter of the simulation (exp distribution for infection)
gamma = float which is the gamma parameter of the simulation (exp distribution for recovery)
day = int indicating the time stamps at which we should print the metrics 
sim_nb = int indicating the max number of simulations
"""
def compute_mean_susceptible(nodes, edges, budget, strategy, time=100, beta=10.0, gamma=0.1, day=30, sim_nb=3):
    assert(time > 0 and beta > 0 and gamma > 0 and budget > 0 and sim_nb > 0 and day > 0 and day <= time and (strategy=="random" or strategy=="vaccination" or strategy=="betweenness" or strategy=="community"))
    day_status = np.zeros(3) # initiate the population status at the given day as a numpy array of dim 3 filled with 0s
    purified_edges = edges.copy()
    if strategy == "vaccination": # if there is a single simulation we have only an axis and not a list of axis thus for implementation reason we wrap it into a list
        purified_edges = cut_edges_max_degree(edges, nodes, budget)
    elif strategy == "betweenness": # if the strategy is betweenness we cut edges with high betweenness
        edges_btw = extract_max_centered_edges()
        for j in range(budget):
            purified_edges.remove(edges_btw[j])
    elif strategy == "community": # if the strategy is community, we locate and isolate the communities in our network
        must_cut_edges = retrieve_community_boundary_edges(nodes, edges)
        if budget < len(must_cut_edges):
            must_cut_edges = must_cut_edges[:budget]
        purified_edges = list(set(edges) - set(must_cut_edges))
    for i in range(sim_nb):  
        if strategy == "random": # if the strategy is random, we shuffle the list of edges and keep only the required number
            purified_edges = edges.copy()  # we put the condition in the loop because we want the simulation to be each time different to have a better "idea" due to randomness
            np.random.shuffle(purified_edges)
            purified_edges = purified_edges[:len(edges)-budget]
        # create the graph for the simulation
        G_sim = nx.Graph()
        G_sim.add_nodes_from(nodes)
        G_sim.add_edges_from(purified_edges)
        # simulate the epidemy
        SIR = epidemics_helper.SimulationSIR(G_sim, beta, gamma)
        SIR.launch_epidemic(source=np.random.randint(0,len(nodes)), max_time=time)
        status_list, nodes_status_count = nodes_status_over_time(SIR, time, [day])
        # update the population status at the give day
        day_status += nodes_status_count[day]
    day_status = day_status / (sim_nb * len(nodes)) # return a percentage of the susceptible person at the given day
    return day_status[0]

import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np 
import epidemics_helper
import pandas as pd

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

def nodes_status_over_time(SIR, time, time_stamps):
    status_count = []
    time_stamps_status_list = {}
    for t in range(time):
        status_list = []
        for node in range(SIR.n_nodes):
            status_list.append(SIR.get_node_status(node,t))
        count = np.bincount(np.asarray(status_list))
        status_count.append(count if count.shape[0] == SIR.STATE_SPACE else np.append(count, np.zeros(len(SIR.STATE_SPACE) - count.shape[0], dtype=int)).tolist())
        for t_s in time_stamps:
            if t == t_s:
                time_stamps_status_list[t] = status_list.copy()
    return time_stamps_status_list, np.asarray(status_count)    

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


def epidemic_markers(nodes_status, marker):
    status = nodes_status/nodes_status[0].sum()
    susceptible = np.where(status[:,[0]] <= marker)[0][0] if np.where(status[:,[0]] <= marker)[0].size > 0 else -1
    infected = np.where(status[:,[1]] >= marker)[0][0] if np.where(status[:,[1]] >= marker)[0].size > 0  else -1
    recovered = np.where(status[:,[2]] >= marker)[0][0] if np.where(status[:,[2]] >= marker)[0].size > 0  else -1
    print(f"The epidemic has the following evolution : {marker*100}% of the population stop being susceptible at time {susceptible}, ", end="")
    print(f"is infected at time {infected} and has recover at time {recovered}")


def extract_max_centered_edges(path="../data/betweenness_edges.csv"):
    df_max_centered_edges = pd.read_csv("../data/max_centered_edges.csv", index_col="Unnamed: 0")
    return [tuple(map(lambda x : int(x),x[1:-1].split(","))) for x in df_max_centered_edges["Edge"]]
    

def strategy_simulation(nodes, edges, time=100, beta=10.0, gamma=0.1, budget=1000, sim_nb=3, day=30, draw=False, strategy="random"):
    assert(time > 0 and beta > 0 and gamma > 0 and budget > 0 and sim_nb > 0 and day > 0 and day <= time and (strategy=="random" or strategy=="vaccination" or strategy=="betweenness" or strategy=="community"))
    fig1, axs1 = plt.subplots(sim_nb, figsize=(10,2*sim_nb))
    fig1.suptitle('Evolution of cases over time')
    axs2 = None
    if draw:
        fig2, axs2 = plt.subplots(sim_nb, figsize=(400,200*sim_nb))
        fig2.suptitle('Epidemic state of the network', fontsize=300)
    if sim_nb == 1:
        axs1 = [axs1]
        if draw:
            axs2 = [axs2]
    day_status = np.zeros(3)
    purified_edges = edges.copy()
    if strategy == "vaccination":
        purified_edges = cut_edges_max_degree(edges, nodes, budget)
    elif strategy == "betweenness":
        edges_btw = extract_max_centered_edges()
        for j in range(budget):
            purified_edges.remove(edges_btw[j])
    elif strategy == "community":
        must_cut_edges = retrieve_community_boundary_edges(nodes, edges)
        if budget < len(must_cut_edges):
            must_cut_edges = must_cut_edges[:budget]
        purified_edges = list(set(edges) - set(must_cut_edges))
    for i in range(sim_nb):
        if strategy == "random":
            purified_edges = edges.copy()
            np.random.shuffle(purified_edges)
            purified_edges = purified_edges[:len(edges)-budget]
        sim_and_draw(nodes, purified_edges, time, beta, gamma, day, day_status, draw, i, sim_nb, axs1, axs2=axs2)
    day_status = day_status / sim_nb
    print(f"By following strategy {strategy}, on day {day}, the average of susceptible people is {day_status[0]}, of infected is {day_status[1]} and of recovered is {day_status[2]}")

def strategy_1_simulation(nodes, edges, time=100, beta=10.0, gamma=0.1, budget=1000, sim_nb=3, day=30, draw=False):
    fig1, axs1 = plt.subplots(sim_nb, figsize=(10,2*sim_nb))
    fig1.suptitle('Evolution of cases over time')
    axs2 = None
    if draw:
        fig2, axs2 = plt.subplots(sim_nb, figsize=(400,200*sim_nb))
        fig2.suptitle('Epidemic state of the network', fontsize=300)
    if sim_nb == 1:
        axs1 = [axs1]
        if draw:
            axs2 = [axs2]
    
    status_on_day_x = np.zeros(3)
    # we execute sim_nb simulations and for each we compute the nodes status over time, plot them and store the points of interest for our statistics on the provided day
    for i in range(sim_nb):
        #we randomly remove budget edges
        purified_edges = edges.copy()
        np.random.shuffle(purified_edges)
        purified_edges = purified_edges[:len(edges)-budget]
        # we create a graph based on this new set of edges and the original set of nodes
        # we simulate an epidemy
        sim_and_draw(nodes, purified_edges, time, beta, gamma, day, status_on_day_x, draw, i, sim_nb, axs1, axs2=axs2)
    status_on_day_x = status_on_day_x / sim_nb
    print(f"On day {day}, the average of susceptible people is {status_on_day_x[0]}, of infected is {status_on_day_x[1]} and of recovered is {status_on_day_x[2]}")
    

def strategy_2_simulation(nodes, edges, edges_btw, time=100, beta=10.0, gamma=0.1, budget=2500, sim_nb=3, day=30, draw=False):
    fig1, axs1 = plt.subplots(sim_nb, figsize=(10,2*sim_nb))
    fig1.suptitle('Evolution of cases over time')
    axs2 = None
    if draw:
        fig2, axs2 = plt.subplots(sim_nb, figsize=(400,200*sim_nb))
        fig2.suptitle('Epidemic state of the network', fontsize=300)
    if sim_nb == 1:
        axs1 = [axs1]
        if draw:
            axs2 = [axs2]
    status_on_day_x = np.zeros(3)
    # we execute sim_nb simulations and for each we compute the nodes status over time, plot them and store the points of interest for our statistics on the provided day
    purified_edges = edges.copy()
    for j in range(budget):
        purified_edges.remove(edges_btw[j])
    for i in range(sim_nb):
        sim_and_draw(nodes, purified_edges, time, beta, gamma, day, status_on_day_x, draw, i, sim_nb, axs1, axs2=axs2)
    status_on_day_x = status_on_day_x / (sim_nb * len(nodes))
    print(f"On day {day}, the average of susceptible people is {status_on_day_x[0]}, of infected is {status_on_day_x[1]} and of recovered is {status_on_day_x[2]}")


def strategy_3_simulation(nodes, edges, time=100, beta=10.0, gamma=0.1, budget=2500, sim_nb=3, day=30, draw=False):
    fig1, axs1 = plt.subplots(sim_nb, figsize=(10,2*sim_nb))
    fig1.suptitle('Evolution of cases over time')
    axs2 = None
    if draw:
        fig2, axs2 = plt.subplots(sim_nb, figsize=(400,200*sim_nb))
        fig2.suptitle('Epidemic state of the network', fontsize=300)
    if sim_nb == 1:
        axs1 = [axs1]
        if draw:
            axs2 = [axs2]
    status_on_day_x = np.zeros(3)
    # we execute sim_nb simulations and for each we compute the nodes status over time, plot them and store the points of interest for our statistics on the provided day
    purified_edges = cut_edges_max_degree(edges, nodes, budget)
    for i in range(sim_nb):
        sim_and_draw(nodes, purified_edges, time, beta, gamma, day, status_on_day_x, draw, i, sim_nb, axs1, axs2=axs2)
    status_on_day_x = status_on_day_x / (sim_nb * len(nodes))
    print(f"On day {day}, the average of susceptible people is {status_on_day_x[0]}, of infected is {status_on_day_x[1]} and of recovered is {status_on_day_x[2]}")


def strategy_4_simulation(nodes, edges, time=100, beta=10.0, gamma=0.1, budget=2500, sim_nb=3, day=30, draw=False):
    fig1, axs1 = plt.subplots(sim_nb, figsize=(10,2*sim_nb))
    fig1.suptitle('Evolution of cases over time')
    axs2 = None
    if draw:
        fig2, axs2 = plt.subplots(sim_nb, figsize=(400,200*sim_nb))
        fig2.suptitle('Epidemic state of the network', fontsize=300)
    if sim_nb == 1:
        axs1 = [axs1]
        if draw:
            axs2 = [axs2]
    status_on_day_x = np.zeros(3)
    # we execute sim_nb simulations and for each we compute the nodes status over time, plot them and store the points of interest for our statistics on the provided day
    must_cut_edges = retrieve_community_boundary_edges(nodes, edges)
    if budget < len(must_cut_edges):
        must_cut_edges = must_cut_edges[:budget]
    purified_edges = list(set(edges) - set(must_cut_edges))
    for i in range(sim_nb):
        sim_and_draw(nodes, purified_edges, time, beta, gamma, day, status_on_day_x, draw, i, sim_nb, axs1, axs2=axs2)
    status_on_day_x = status_on_day_x / (sim_nb * len(nodes))
    print(f"On day {day}, the average of susceptible people is {status_on_day_x[0]}, of infected is {status_on_day_x[1]} and of recovered is {status_on_day_x[2]}")


def retrieve_community_boundary_edges(nodes, edges):
    G_init = nx.Graph()
    G_init.add_nodes_from(nodes)
    G_init.add_edges_from(edges)
    communities = nx.algorithms.community.greedy_modularity_communities(G_init)
    list_must_cut_edges = []
    for i in range(len(communities)):
        for j in range(len(communities)):
            if j != i:
                list_must_cut_edges += [e for e in nx.edge_boundary(G_init, communities[i], nbunch2=communities[j])]
    list_must_cut_edges = list(set(list_must_cut_edges) - (set(list_must_cut_edges) -set(edges)))
    return list_must_cut_edges


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

def cut_edges_max_degree(edges, nodes, budget):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    degrees_sorted = sorted(list(G.degree()), key = lambda t : t[1], reverse=True)
    new_edges_list = edges.copy()
    cost = 0
    for node in degrees_sorted:
        node_to_kill = node[0]
        for edge in new_edges_list:
            if(cost >= budget):
                return new_edges_list
            if edge[0] == node_to_kill or edge[1] == node_to_kill :
                new_edges_list.remove(edge)
                cost+=1
    return new_edges_list


def sim_and_draw(nodes, purified_edges, time, beta, gamma, day, day_status, draw, sim_id, sim_nb, axs1, axs2=None):
        G_sim = nx.Graph()
        G_sim.add_nodes_from(nodes)
        G_sim.add_edges_from(purified_edges)
        # we simulate an epidemy
        SIR = epidemics_helper.SimulationSIR(G_sim, beta, gamma)
        SIR.launch_epidemic(source=np.random.randint(0,len(nodes)), max_time=time)
        status_list, nodes_status_count = nodes_status_over_time(SIR, time, [day])
        if sim_nb != 1:
            axs1[sim_id].set_title("Simulation "+str(sim_id+1))
        plot_population_status(nodes_status_count, ax=axs1[sim_id], percentage=True)
        if draw:
            if sim_nb != 1:
                axs2[sim_id].set_title("Simulation "+str(sim_id+1), fontsize=300)
            draw_graph(G_sim, nodes_status=status_list[day], ax=axs2[sim_id])
        day_status += nodes_status_count[day]

def compute_mean_susceptible(nodes, edges, budget, strategy, time=100, beta=10.0, gamma=0.1, day=30, sim_nb=3):
    assert(time > 0 and beta > 0 and gamma > 0 and budget > 0 and sim_nb > 0 and day > 0 and day <= time and (strategy=="random" or strategy=="vaccination" or strategy=="betweenness" or strategy=="community"))
    day_status = np.zeros(3)
    purified_edges = edges.copy()
    if strategy == "vaccination":
        purified_edges = cut_edges_max_degree(edges, nodes, budget)
    elif strategy == "betweenness":
        edges_btw = extract_max_centered_edges()
        for j in range(budget):
            purified_edges.remove(edges_btw[j])
    elif strategy == "community":
        must_cut_edges = retrieve_community_boundary_edges(nodes, edges)
        if budget < len(must_cut_edges):
            must_cut_edges = must_cut_edges[:budget]
        purified_edges = list(set(edges) - set(must_cut_edges))
    for i in range(sim_nb):
        if strategy == "random":
            purified_edges = edges.copy()
            np.random.shuffle(purified_edges)
            purified_edges = purified_edges[:len(edges)-budget]
        G_sim = nx.Graph()
        G_sim.add_nodes_from(nodes)
        G_sim.add_edges_from(purified_edges)
        SIR = epidemics_helper.SimulationSIR(G_sim, beta, gamma)
        SIR.launch_epidemic(source=np.random.randint(0,len(nodes)), max_time=time)
        status_list, nodes_status_count = nodes_status_over_time(SIR, time, [day])
        day_status += nodes_status_count[day]
    day_status = day_status / (sim_nb * len(nodes))
    return day_status[0]

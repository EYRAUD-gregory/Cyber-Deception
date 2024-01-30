from modelling import Modelling

if __name__ == '__main__':

    model = Modelling(M=5, K=15)

    G = model.create_graph()

    model.plot_graph()
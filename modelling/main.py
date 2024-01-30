from modelling import Modelling

if __name__ == '__main__':

    model = Modelling(M=10, n=3)

    G = model.create_graph()

    model.plot_graph()
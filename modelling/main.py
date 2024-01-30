from modelling import Modelling

if __name__ == '__main__':

    model = Modelling(M=8, n=5)

    G = model.create_graph()

    model.plot_graph()
# Simulating Social Media Networks as Content Markets

This repository implements a simulation of a social media community as a content market. It was produced in collaboration with Professor Peter Marbach as part of a CSC494 course at the University of Toronto. The simulation is based on the formalization defined in Su & Marbach (2023).

An online social network is viewed as a content market with two types of members. Periphery members produce and consume content related to their interests, while core members (influencers) collect produced content and reshare it to periphery members. Periphery members allocate following to other periphery members and to the influencer up to a set attention bound. They receive utility for consuming content related to their main interests and for producing content that attracts them a large following from the influencer and other periphery members. The influencer receives utility for attracting a large following of periphery members.

The utilities defined here constitute a set of optimization problems that can be iteratively solved until the market reaches equilibrium. The cited paper provides several conditions for this equilibrium and discusses the effects of market parameters. The goal of this repository is to simulate the market optimization and validate the theoretical propositions of the model.

Su, J., Marbach, P. (2023). Structure of Core-Periphery Communities. In: Cherifi, H., Mantegna, R.N., Rocha, L.M., Cherifi, C., Micciche, S. (eds) Complex Networks and Their Applications XI. COMPLEX NETWORKS 2016 2022. Studies in Computational Intelligence, vol 1078. Springer, Cham. https://doi.org/10.1007/978-3-031-21131-7\_12

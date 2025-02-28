from game import Game, SevenCardStud


class SamplingScheme():

    def __init__(self, obj: Game):
        if not isinstance(obj, Game):
            raise TypeError("Must call a Stud subclass when creating an instance")
        
        self.obj = obj
        self.timestep = 0
        self.nodes_traversed = 0

    def alternating_outcome_sampling(self, prefix, update_player, timestep, reach_update_player, reach_opponent, sampling_probability):
        
        self.nodes_traversed += 1

        if self.obj.is_terminal(prefix):
            return self.obj.utility_function(prefix)
        elif self.obj.turn_to_act(prefix) == -1:
            # sample new action
            # prefix.append(new_action)
            return self.alternating_outcome_sampling(self, prefix, update_player, timestep, reach_update_player, reach_opponent, sampling_probability)
        
        # InformationSet.node(prefix)
        # regret_matching(InformationSet.node(prefix))
        
    def lazy_weighted_outcome_sampling(self, prefix, update_player, timestep, reach_update_player, reach_opponent, sampling_probability, weight_1, weight_2):
        pass

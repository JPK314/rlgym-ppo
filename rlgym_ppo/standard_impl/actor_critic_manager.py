from rlgym_ppo.api import ActorCriticManager


class BasicActorCriticManager(ActorCriticManager[int]):
    def should_discard(self, agent_id):
        return False

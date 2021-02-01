from locust import HttpLocust, TaskSet, task
import json


class UserBehavior(TaskSet):
    @task(1)
    def create_post(self):
        headers = {'content-type': 'application/x-www-form-urlencoded; charset=UTF-8', 'Accept-Encoding': 'gzip'}
        self.client.post("/seedwords2",
                         data=json.dumps({
                             'algorithm': 'Algorithm: Linear projection',
                             'concept1_name': 'Gender',
                             'concept2_name': 'Concept2',
                             'equalize': 'man-woman,he-him,she-her',
                             'evalwords': 'engineer, lawyer, receptionist, homemaker',
                             'orth_subspace': 'scientist, doctor, nurse, secretary, maid, dancer, cleaner, advocate, player, banker',
                             'seedwords1': 'he',
                             'seedwords2': 'she',
                             'subspace_method': 'Subspace method: Two means'}),
                         headers=headers, name="Create a new post")


class WebsiteUser(HttpLocust):
    task_set = UserBehavior


class QuickstartUser(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def make_request(self):
        self.client.post("seedwords2", body={
            'algorithm': 'Algorithm: Linear projection',
            'concept1_name': 'Gender',
            'concept2_name': 'Concept2',
            'equalize': 'man-woman,he-him,she-her',
            'evalwords': 'engineer, lawyer, receptionist, homemaker',
            'orth_subspace': 'scientist, doctor, nurse, secretary, maid, dancer, cleaner, advocate, player, banker',
            'seedwords1': 'he',
            'seedwords2': 'she',
            'subspace_method': 'Subspace method: Two means'
        })

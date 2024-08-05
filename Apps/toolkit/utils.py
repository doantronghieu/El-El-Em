import uuid
from faker import Faker
from typing import Literal

def generate_unique_id(
	thing: Literal["uuid", "name", "uuid_name"]
):
	random_uuid = str(uuid.uuid4())
	random_name = Faker().name()
	random_uuid_name = f"{random_name}-{random_uuid}"
	
	if thing == "uuid":
		return random_uuid
	elif thing == "name":
		return random_name
	elif thing == "uuid_name":
		return random_uuid_name
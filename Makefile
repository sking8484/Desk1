include .env

build-image:
	docker compose build

create-ecr-repo: build-image
	-aws ecr create-repository --repository-name $(ECR_REPO_NAME)

publish: create-ecr-repo
	echo $(AWS_ACCESS_KEY_ID)
	docker tag deskonecontainer:latest $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$(ECR_REPO_NAME):latest 
	aws ecr get-login-password | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$(ECR_REPO_NAME):latest

deploy: publish
	aws cloudformation deploy --stack-name $(CFN_STACK_NAME) \
	--template-file ./templateFile.yml \
	--parameter-overrides imageUri=$(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$(ECR_REPO_NAME):latest

tear-down:
	aws cloudformation delete-stack --stack-name $(CFN_STACK_NAME)

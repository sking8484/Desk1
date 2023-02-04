include .env
# Export the service list
export SERVICE_LIST=./deployment/service-list.txt

test-imports:
	for service in `cat $${SERVICE_LIST}`; do \
		python3 src/$${service}/handler.py ; \
	done 

test: 
ifeq ($(TEST_FILE), )
	python3 -W ignore:PendingDeprecationWarning -m unittest discover -s src -vvv -f
else
	python3 -W ignore:PendingDeprecationWarning -m unittest discover -s src -vvv -p $(TEST_FILE) -f
endif

build-images: test
	for service in `cat $${SERVICE_LIST}`; do \
		rm -rf ./deployment/$${service} ; \
		mkdir ./deployment/$${service} ; \
		cp Dockerfile ./deployment/$${service}/ ; \
		docker build -f "./deployment/$${service}/." -t "$${service}-container" . --build-arg function=$${service} ; \
	done
	#docker compose build

create-ecr-repo: build-images
	for service in `cat $${SERVICE_LIST}`; do \
		aws ecr create-repository --repository-name $${service}-repo || true ; \
	done

publish: create-ecr-repo
	for service in `cat $${SERVICE_LIST}`; do \
		docker tag $${service}-container:latest $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:latest ; \
		aws ecr get-login-password | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com ; \
		docker push $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:latest ; \
	done

deploy: publish
	aws cloudformation deploy --stack-name $(CFN_STACK_NAME) \
	--template-file ./templateFile.yml \
	--parameter-overrides imageUri=$(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$(ECR_REPO_NAME):latest

tear-down:
	aws cloudformation delete-stack --stack-name $(CFN_STACK_NAME)

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

build-images: 
	for service in `cat $${SERVICE_LIST}`; do \
		rm -rf ./deployment/$${service} ; \
		mkdir ./deployment/$${service} ; \
		cp Dockerfile ./deployment/$${service}/ ; \
		docker build -f "./deployment/$${service}/Dockerfile" -t "$${service}-container" . --build-arg function=$${service} ; \
	done
	#docker compose build

create-ecr-repo: build-images
	for service in `cat $${SERVICE_LIST}`; do \
		aws ecr create-repository --repository-name $${service}-repo || true ; \
	done

publish: create-ecr-repo
	for service in `cat $${SERVICE_LIST}`; do \
		docker tag $${service}-container:latest $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:${TAG} ; \
		aws ecr get-login-password | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com ; \
		docker push $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:${TAG} ; \
	done

deploy:
	sam deploy --stack-name infra-$(CFN_STACK_NAME) \
	--template-file ./templateFileInfra.yml --capabilities CAPABILITY_IAM

deployLambdas: publish
	for service in `cat $${SERVICE_LIST}`; do \
		sam deploy --stack-name $${service}-$(CFN_STACK_NAME) \
		--template-file ./templateFileLambdas.yml --image-repository $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo \
		--parameter-overrides imageUri=$(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$${service}-repo:${TAG} DBUSER=${DBUSER} DBPASSWORD=${DBPASSWORD}\
		DBHOST=${DBHOST} DBPORT=${DBPORT} DBNAME=${DBNAME} MAINSTOCKTABLE=${MAINSTOCKTABLE}\
		MAINPREDICTIONTABLE=${MAINPREDICTIONTABLE} MAINWEIGHTSTABLE=${MAINWEIGHTSTABLE} MAINGERBERTABLE=${MAINGERBERTABLE}\
		ALPACAPUBKEY=${ALPACAPUBKEY} ALPACAPRIVKEY=${ALPACAPRIVKEY} Service=$${service} ENV=${ENV} MAINPERFTABLE=${MAINPERFTABLE} MAINFACTORTABLE=${MAINFACTORTABLE};\
	done

destroy:
	for service in `cat $${SERVICE_LIST}`; do \
		aws cloudformation delete-stack --stack-name $${service}-$(CFN_STACK_NAME); \
	done

start-database:
	docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=test -d -v mysql:/var/lib/mysql -p 3307:3306 mysql

destroy-database:
	docker rm -f some-mysql

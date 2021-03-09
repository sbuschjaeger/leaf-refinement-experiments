GROUP=$(shell id --group --name)
GID=$(shell id --group)
USER=$(shell id --user --name)
UID=$(shell id --user)
# GITURL=$(shell git config --get remote.origin.url)
# GITBRANCH=$(shell git rev-parse --abbrev-ref HEAD)
# GITCOMMIT=$(shell git rev-parse --verify HEAD)
CONTAINER_NAME=hep

# constrain the build node: make BUILD_ARGS="--build-arg constraint:node==s876cn03"
BUILD_ARGS=

# Download all datasets and code from git repos
.init: 
	for dataset in adult bank connect covtype dry-beans eeg elec gas-drift japanese-vowels letter magic mozilla mushroom pen-digits satimage shuttle spambase thyroid wine-quality ; do \
		cd %%dataset; \
		./init.sh; \
		cd ..
	done
	# git clone git@github.com:sbuschjaeger/experiment_runner.git
	# git clone git@github.com:sbuschjaeger/PyPruning.git

#--build-arg giturl=$(GITURL) 
image: .IMAGE
.IMAGE: Dockerfile .code
	- docker rmi -f $(USER)/$(CONTAINER_NAME)
	docker build \
	    --build-arg group=$(GROUP) \
	    --build-arg gid=$(GID) \
	    --build-arg user=$(USER)-$(CONTAINER_NAME) \
	    --build-arg uid=$(UID) \
	    --tag $(USER)/$(CONTAINER_NAME) \
	    $(BUILD_ARGS) .
	echo "$(USER)/$(CONTAINER_NAME)" > $@

# .code: ../.git/COMMIT_EDITMSG
# 	git clone ../ .code || ( cd .code && git fetch --all && git checkout $(GITBRANCH) && git pull && cd .. )

# push to $DOCKER_REPOSITORY, but only if this variable is set
push: .PUSH
.PUSH: .IMAGE
ifndef DOCKER_REPOSITORY
	$(error $$DOCKER_REPOSITORY is not set)
else
	- docker rmi -f $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	docker tag $(USER)/$(CONTAINER_NAME) $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	docker push $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	docker pull $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	echo "$(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)" > $@
endif

clean:
	- docker rmi -f $(USER)/$(CONTAINER_NAME)
	- docker rmi -f $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	rm -f .IMAGE .PUSH .requirements.txt
	rm -rf .code

.PHONY: image push clean
SHELL := /bin/bash

update:
	GOPROXY=direct go get github.com/itsubaki/autograd@HEAD
	go get -u ./...
	go mod tidy

test:
	go test -cover $(shell go list ./... | grep -v /cmd/ ) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

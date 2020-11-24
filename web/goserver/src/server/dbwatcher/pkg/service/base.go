package service

import (
	"context"

	"go.mongodb.org/mongo-driver/mongo"

	kitendpoint "server/kit/endpoint"
)

type DatabaseWatcherService interface {
	Problem(ctx context.Context, req ProblemsWatcherRequestData, responseChan chan kitendpoint.Response)
}

type basicDatabaseWatcherService struct {
	db *mongo.Database
}

// NewBasicApiService returns a naive, stateless implementation of ApiService.
func NewBasicDatabaseWatcherService(db *mongo.Database) DatabaseWatcherService {
	return &basicDatabaseWatcherService{db}
}

// New returns a ApiService with all of the expected middleware wired in.
func New(db *mongo.Database, middleware []Middleware) DatabaseWatcherService {
	var svc = NewBasicDatabaseWatcherService(db)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}

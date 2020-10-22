package service

import (
	"context"

	"go.mongodb.org/mongo-driver/mongo"

	t "server/db/pkg/types"
)

type DatabaseService interface {
	AssetFind(ctx context.Context, req AssetFindRequestData) t.AssetFindResponse
	AssetFindOne(ctx context.Context, req AssetFindOneRequestData) t.Asset
	AssetUpdateUpsert(ctx context.Context, req AssetUpdateUpsertRequestData) t.Asset

	BuildFind(ctx context.Context, req BuildFindRequestData) t.BuildFindResponse
	BuildFindOne(ctx context.Context, req BuildFindOneRequestData) t.Build
	BuildInsertOne(ctx context.Context, req BuildInsertOneRequestData) t.Build
	BuildUpdateOne(ctx context.Context, req BuildUpdateOneRequestData) t.Build

	CvatTaskFind(ctx context.Context, req CvatTaskFindRequestData) (CvatTaskFindResponse, error)
	CvatTaskFindOne(ctx context.Context, req CvatTaskFindOneRequestData) t.CvatTask
	CvatTaskInsertOne(ctx context.Context, req CvatTaskInsertOneRequestData) t.CvatTask
	CvatTaskUpdateOne(ctx context.Context, req CvatTaskUpdateOneRequestData) t.CvatTask

	ProblemDelete(ctx context.Context, req ProblemDeleteRequestData) ProblemDeleteResponseData
	ProblemFind(ctx context.Context, req ProblemFindRequestData) t.ProblemFindResponse
	ProblemFindOne(ctx context.Context, req ProblemFindOneRequestData) t.Problem
	ProblemUpdateUpsert(ctx context.Context, req ProblemUpdateUpsertRequestData) t.Problem

	ModelFind(ctx context.Context, req ModelFindRequestData) t.ModelFindResponse
	ModelFindOne(ctx context.Context, req ModelFindOneRequestData) t.Model
	ModelInsertOne(ctx context.Context, req ModelInsertOneRequestData) (t.Model, error)
	ModelUpdateOne(ctx context.Context, req ModelUpdateOneRequestData) t.Model
	ModelUpdateUpsert(ctx context.Context, req ModelUpdateUpsertRequestData) t.Model
}

type basicDatabaseService struct {
	db *mongo.Database
}

// NewBasicApiService returns a naive, stateless implementation of ApiService.
func NewBasicDatabaseService(db *mongo.Database) DatabaseService {
	return &basicDatabaseService{db}
}

// New returns a ApiService with all of the expected middleware wired in.
func New(db *mongo.Database, middleware []Middleware) DatabaseService {
	var svc = NewBasicDatabaseService(db)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}

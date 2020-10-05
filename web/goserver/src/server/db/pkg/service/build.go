package service

import (
	"context"
	"errors"
	"log"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	n "server/common/names"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
)

type BuildFindRequestData struct {
	Page      int64              `json:"page" bson:"page"`
	Size      int64              `json:"size" bson:"size"`
	ProblemId primitive.ObjectID `json:"problemId" bson:"problemId"`
	Status    string             `json:"status" bson:"status"`
}

func (s *basicDatabaseService) BuildFind(ctx context.Context, req BuildFindRequestData) (result t.BuildFindResponse) {
	c := s.db.Collection(n.CBuild)
	option := options.Find()
	option.SetSkip(req.Size * (req.Page - 1))
	option.SetLimit(req.Size)
	filter := bson.M{}
	if !req.ProblemId.IsZero() {
		filter["problemId"] = req.ProblemId
	}
	if req.Status != "" {
		filter["status"] = req.Status
	}
	total, err := c.CountDocuments(ctx, filter, options.Count())
	if err != nil {
		return t.BuildFindResponse{BaseList: t.BaseList{}}
	}
	var items []t.Build
	cur, err := c.Find(ctx, filter, option)
	if err != nil {
		return t.BuildFindResponse{BaseList: t.BaseList{Total: total}, Items: items}
	}
	for cur.Next(context.TODO()) {
		// create a value into which the single document can be decoded
		var elem t.Build
		err := cur.Decode(&elem)
		if err != nil {
			return t.BuildFindResponse{BaseList: t.BaseList{}}
		}

		items = append(items, elem)
	}

	if err := cur.Err(); err != nil {
		return t.BuildFindResponse{BaseList: t.BaseList{}}
	}

	// Close the cursor once finished
	cur.Close(context.TODO())

	return t.BuildFindResponse{BaseList: t.BaseList{Total: total}, Items: items}
}

type BuildFindOneRequestData struct {
	Id        primitive.ObjectID `json:"id" bson:"_id"`
	ProblemId primitive.ObjectID `json:"problemId" bson:"problemId"`
	Status    string             `json:"status" bson:"status"`
	Name      string             `json:"name" bson:"name"`
}

func (s *basicDatabaseService) BuildFindOne(ctx context.Context, req BuildFindOneRequestData) (result t.Build) {
	buildCollection := s.db.Collection(n.CBuild)
	var filter interface{}
	if primitive.ObjectID.IsZero(req.Id) {
		if req.Status == buildStatus.Tmp {
			filter = bson.M{"problemId": req.ProblemId, "status": req.Status}
		} else if req.Name != "" {
			filter = bson.M{"problemId": req.ProblemId, "name": req.Name}
		}
	} else {
		filter = bson.M{"_id": req.Id}
	}
	err := buildCollection.FindOne(ctx, filter).Decode(&result)
	if err != nil {
		log.Println("BuildFindOne.buildCollection.FindOne(ctx, filter).Decode(&result)", err)
	}
	return
}

type BuildInsertOneRequestData struct {
	ProblemId primitive.ObjectID            `bson:"problemId" json:"problemId"`
	Folder    string                        `bson:"folder" json:"folder"`
	Name      string                        `bson:"name" json:"name"`
	Split     map[string]t.BuildAssetsSplit `bson:"split" json:"split"`
	Status    string                        `bson:"status" json:"status"`
}

func (s *basicDatabaseService) BuildInsertOne(ctx context.Context, req BuildInsertOneRequestData) (result t.Build) {
	buildCollection := s.db.Collection(n.CBuild)
	r, err := buildCollection.InsertOne(ctx, req)
	if err != nil {
		log.Println("BuildInsertOne", err)
		if IsDup(err) {
			err = buildCollection.FindOne(ctx, bson.M{"problemId": req.ProblemId, "name": req.Name, "status": req.Status}).Decode(&result)
		}
		return
	}
	err = buildCollection.FindOne(ctx, bson.M{"_id": r.InsertedID}).Decode(&result)
	return
}

func IsDup(err error) bool {
	var e mongo.WriteException
	if errors.As(err, &e) {
		for _, we := range e.WriteErrors {
			if we.Code == 11000 {
				return true
			}
		}
	}
	return false
}

type BuildUpdateOneRequestData = t.Build

func (s *basicDatabaseService) BuildUpdateOne(ctx context.Context, req BuildUpdateOneRequestData) (result t.Build) {
	buildCollection := s.db.Collection(n.CBuild)
	option := options.Update()
	_, err := buildCollection.UpdateOne(
		ctx,
		bson.M{"_id": req.Id},
		bson.M{"$set": req},
		option,
	)
	if err != nil {
		log.Print("BuildUpdateOne.UpdateOne", err)
	}
	err = buildCollection.FindOne(ctx, bson.M{"_id": req.Id}).Decode(&result)
	if err != nil {
		log.Print("BuildUpdateOne.FindOne", err)
	}
	return
}

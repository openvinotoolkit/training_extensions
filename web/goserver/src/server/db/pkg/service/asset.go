package service

import (
	"context"
	"log"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"

	n "server/common/names"
	t "server/db/pkg/types"
)

type AssetFindRequestData struct {
	ParentFolder string `bson:"parentFolder" json:"parentFolder"`
	Page         int64  ` bson:"page" json:"page"`
	Size         int64  ` bson:"size" json:"size"`
}

func (s *basicDatabaseService) AssetFind(ctx context.Context, req AssetFindRequestData) (result t.AssetFindResponse) {
	assetCollection := s.db.Collection(n.CAsset)
	option := options.Find()
	if req.Page > 0 && req.Size > 0 {
		option.SetSkip(req.Size * (req.Page - 1))
		option.SetLimit(req.Size)
	}
	total, err := assetCollection.CountDocuments(ctx, bson.M{"parentFolder": req.ParentFolder}, options.Count())
	cur, err := assetCollection.Find(ctx, bson.M{"parentFolder": req.ParentFolder}, option)
	var items []t.Asset
	if err != nil {
		return t.AssetFindResponse{
			Items: items,
		}
	}
	for cur.Next(context.TODO()) {
		// create a value into which the single document can be decoded
		var elem t.Asset
		err := cur.Decode(&elem)
		if err != nil {
			return t.AssetFindResponse{BaseList: t.BaseList{}}
		}
		items = append(items, elem)
	}
	if err := cur.Err(); err != nil {
		return t.AssetFindResponse{BaseList: t.BaseList{}}
	}
	cur.Close(context.TODO())
	log.Println(items)

	return t.AssetFindResponse{BaseList: t.BaseList{Total: total}, Items: items}

}

type AssetFindOneRequestData struct {
	Id           primitive.ObjectID `bson:"_id" json:"id"`
	ParentFolder string             `bson:"parentFolder" json:"parentFolder"`
	Name         string             `bson:"name" json:"name"`
}

func (s *basicDatabaseService) AssetFindOne(ctx context.Context, req AssetFindOneRequestData) (result t.Asset) {
	log.Println("AssetFindOne")
	assetCollection := s.db.Collection(n.CAsset)
	filter := bson.M{}
	if !primitive.ObjectID.IsZero(req.Id) {
		filter = bson.M{"_id": req.Id}
	} else if req.Name != "" {
		filter = bson.M{"parentFolder": req.ParentFolder, "name": req.Name}
	}
	err := assetCollection.FindOne(ctx, filter).Decode(&result)
	if err != nil {
		log.Print("AssetFindOne", err)
	}
	return
}

type AssetUpdateUpsertRequestData struct {
	ParentFolder string `bson:"parentFolder" json:"parentFolder"`
	Name         string `bson:"name" json:"name"`
	Type         string `bson:"type" json:"type"`
	CvatDataPath string `bson:"cvatDataPath" json:"cvatDataPath"`
}

func (s *basicDatabaseService) AssetUpdateUpsert(ctx context.Context, req AssetUpdateUpsertRequestData) (result t.Asset) {
	log.Println("AssetUpdateUpsert")
	assetCollection := s.db.Collection(n.CAsset)
	option := options.Update()
	option.SetUpsert(true)
	_, err := assetCollection.UpdateOne(
		ctx,
		bson.M{
			"name":         req.Name,
			"parentFolder": req.ParentFolder,
		},
		bson.D{{"$set", req}},
		option,
	)
	if err != nil {
		log.Println("UpdateOne", err)
	}
	err = assetCollection.FindOne(
		ctx,
		bson.M{
			"name":         req.Name,
			"parentFolder": req.ParentFolder,
			"type":         req.Type,
		},
	).Decode(&result)
	if err != nil {
		log.Println("assetCollection.FindOne.Decode(&result)", err)
	}
	return result
}

package service

import (
	"context"
	"log"
	"strings"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"go.mongodb.org/mongo-driver/bson/primitive"

	buildInsertOne "server/db/pkg/handler/build/insert_one"
	splitState "server/db/pkg/types/build/split_state"

	assetFind "server/db/pkg/handler/asset/find"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
)

type CreateEmptyRequestData struct {
	ProblemId primitive.ObjectID `bson:"problemId" json:"problemId"`
}

func (s *basicBuildService) CreateEmpty(ctx context.Context, req CreateEmptyRequestData) t.Build {
	split := getEmptyBuildSplit()
	addAssetsToSplit(ctx, s.Conn, &split, ".")
	build := s.insertTmpBuild(req.ProblemId, split)
	return build
}

func (s *basicBuildService) insertTmpBuild(problemId primitive.ObjectID, split map[string]t.BuildAssetsSplit) t.Build {
	buildInsertOneRequestData := buildInsertOne.RequestData{
		ProblemId: problemId,
		Folder:    "__tmp__",
		Name:      "__tmp__",
		Status:    buildStatus.Tmp,
		Split:     split,
	}

	buildInsertOneResp := <-buildInsertOne.Send(
		context.TODO(),
		s.Conn,
		buildInsertOneRequestData,
	)
	return buildInsertOneResp.Data.(buildInsertOne.ResponseData)

}

func getEmptyBuildSplit() map[string]t.BuildAssetsSplit {
	return map[string]t.BuildAssetsSplit{
		".": {
			Children: map[string]t.BuildAssetsSplit{},
			Test:     splitState.Rejected,
			Train:    splitState.Rejected,
			Val:      splitState.Rejected,
		},
	}
}

func addAssetsToSplit(ctx context.Context, conn *rabbitmq.Connection, split *map[string]t.BuildAssetsSplit, path string) {
	assetFindResp := <-assetFind.Send(
		ctx,
		conn,
		assetFind.RequestData{
			ParentFolder: path,
		},
	)
	assets := assetFindResp.Data.(assetFind.ResponseData).Items

	for _, asset := range assets {
		assetPathArr := strings.Split(asset.ParentFolder, "/")
		assetPathArr = append(assetPathArr, asset.Name)
		log.Println(assetPathArr)
		addAssetByPath(split, assetPathArr, asset.Id)
		log.Println(split)
		assetPathString := strings.Join(assetPathArr, "/")
		addAssetsToSplit(ctx, conn, split, assetPathString)
	}
}

func addAssetByPath(split *map[string]t.BuildAssetsSplit, path []string, assetId primitive.ObjectID) {
	buildAssetSplit := (*split)[path[0]]
	for _, folder := range path[1:] {
		if _, ok := buildAssetSplit.Children[folder]; !ok {
			buildAssetSplit.Children[folder] = t.BuildAssetsSplit{
				AssetId:  assetId,
				Children: map[string]t.BuildAssetsSplit{},
				Test:     splitState.Rejected,
				Train:    splitState.Rejected,
				Val:      splitState.Rejected,
			}
		}
		buildAssetSplit = buildAssetSplit.Children[folder]
	}
}

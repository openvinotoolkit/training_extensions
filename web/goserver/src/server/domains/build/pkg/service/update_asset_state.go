package service

import (
	"context"
	"log"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	assetFindOne "server/db/pkg/handler/asset/find_one"
	buildFindOne "server/db/pkg/handler/build/find_one"
	buildUpdateOne "server/db/pkg/handler/build/update_one"
	cvatTaskFindOne "server/db/pkg/handler/cvat_task/find_one"
	t "server/db/pkg/types"
	splitState "server/db/pkg/types/build/split_state"
	buildStatus "server/db/pkg/types/build/status"
)

type UpdateAssetStateRequestData struct {
	Id    primitive.ObjectID `bson:"_id" json:"id"`
	Test  int                `bson:"test" json:"test"`
	Train int                `bson:"train" json:"train"`
	Val   int                `bson:"val" json:"val"`
}

type BuildSplit struct {
	Train int `bson:"train" json:"train"`
	Test  int `bson:"test" json:"test"`
	Val   int `bson:"val" json:"val"`
}

type UpdateAssetStateResponseData struct {
	Train int                `bson:"train" json:"train"`
	Test  int                `bson:"test" json:"test"`
	Val   int                `bson:"val" json:"val"`
	Id    primitive.ObjectID `bson:"_id" json:"id"`
}

func (s *basicBuildService) UpdateAssetState(ctx context.Context, req UpdateAssetStateRequestData) UpdateAssetStateResponseData {
	cvatTaskFindOneResp := <-cvatTaskFindOne.Send(
		ctx,
		s.Conn,
		cvatTaskFindOne.RequestData{
			Id: req.Id,
		},
	)
	cvatTask := cvatTaskFindOneResp.Data.(cvatTaskFindOne.ResponseData)

	assetFindOneResp := <-assetFindOne.Send(
		ctx,
		s.Conn,
		assetFindOne.RequestData{
			Id: cvatTask.AssetId,
		},
	)
	asset := assetFindOneResp.Data.(assetFindOne.ResponseData)

	buildFindOneResp := <-buildFindOne.Send(
		ctx,
		s.Conn,
		buildFindOne.RequestData{
			ProblemId: cvatTask.ProblemId,
			Status:    buildStatus.Tmp,
		},
	)
	build := buildFindOneResp.Data.(buildFindOne.ResponseData)

	path := strings.Split(asset.ParentFolder, "/")
	path = append(path, asset.Name)

	buildSplit := BuildSplit{
		Test:  req.Test,
		Val:   req.Val,
		Train: req.Train,
	}
	build.Split[path[0]] = fixBuildAssetSplitTreeChildren(build.Split[path[0]], buildSplit, path[1:])
	build.Split[path[0]] = fixBuildAssetSplitTreeParents(build.Split[path[0]], path[0])
	<-buildUpdateOne.Send(
		ctx,
		s.Conn,
		build,
	)
	return UpdateAssetStateResponseData{
		Id:    req.Id,
		Test:  req.Test,
		Val:   req.Val,
		Train: req.Train,
	}
}

func fixBuildAssetSplitTreeChildren(node t.BuildAssetsSplit, split BuildSplit, path []string) t.BuildAssetsSplit {
	if len(path) > 0 && len(node.Children) > 0 {
		log.Println(path, node.Children)
		node.Children[path[0]] = fixBuildAssetSplitTreeChildren(node.Children[path[0]], split, path[1:])
		return node
	}
	if split.Train != splitState.Indeterminate {
		node.Train = split.Train
	}
	if split.Test != splitState.Indeterminate {
		node.Test = split.Test
	}
	if split.Val != splitState.Indeterminate {
		node.Val = split.Val
	}
	for name, child := range node.Children {
		node.Children[name] = fixBuildAssetSplitTreeChildren(child, split, path)
	}
	return node
}

func fixBuildAssetSplitTreeParents(parent t.BuildAssetsSplit, n string) t.BuildAssetsSplit {
	for name, child := range parent.Children {
		parent.Children[name] = fixBuildAssetSplitTreeParents(child, name)
	}
	if len(parent.Children) == 0 {
		return parent
	}
	var randKey string
	for randKey, _ = range parent.Children {
		break
	}
	sumUp := BuildSplit{
		Test:  parent.Children[randKey].Test,
		Train: parent.Children[randKey].Train,
		Val:   parent.Children[randKey].Val,
	}

	for _, v := range parent.Children {
		if sumUp.Train != v.Train {
			sumUp.Train = splitState.Indeterminate
		}
		if sumUp.Test != v.Test {
			sumUp.Test = splitState.Indeterminate
		}
		if sumUp.Val != v.Val {
			sumUp.Val = splitState.Indeterminate
		}
	}
	parent.Test = sumUp.Test
	parent.Train = sumUp.Train
	parent.Val = sumUp.Val
	return parent
}

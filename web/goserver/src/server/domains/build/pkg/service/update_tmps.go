package service

import (
	"context"

	buildFind "server/db/pkg/handler/build/find"
	buildUpdateOne "server/db/pkg/handler/build/update_one"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
)

type UpdateTmpsRequestData struct {
}

type UpdateTmpsResponseData struct {
}

func (s *basicBuildService) UpdateTmps(ctx context.Context, req UpdateTmpsRequestData) UpdateTmpsResponseData {
	emptySplit := getEmptyBuildSplit()
	addAssetsToSplit(ctx, s.Conn, &emptySplit, ".")
	tmpBuilds := s.getTmpBuilds()
	for _, tmpBuild := range tmpBuilds {
		tmpBuild.Split["."] = mergeTmpSplits(tmpBuild.Split["."], emptySplit["."])
		s.updateBuild(tmpBuild)
	}
	return UpdateTmpsResponseData{}
}

func (s *basicBuildService) updateBuild(build t.Build) {
	<-buildUpdateOne.Send(context.TODO(), s.Conn, build)
}

func (s *basicBuildService) getTmpBuilds() []t.Build {
	buildFindResp := <-buildFind.Send(context.TODO(), s.Conn, buildFind.RequestData{Status: buildStatus.Tmp})
	return buildFindResp.Data.(buildFind.ResponseData).Items
}

func mergeTmpSplits(from, to t.BuildAssetsSplit) t.BuildAssetsSplit {
	to.Train = from.Train
	to.Val = from.Val
	to.Test = from.Test
	to.CvatTaskAnnotationId = from.CvatTaskAnnotationId
	for fromBuildId, fromChild := range from.Children {
		to.Children[fromBuildId] = mergeTmpSplits(fromChild, to.Children[fromBuildId])
	}
	return to
}

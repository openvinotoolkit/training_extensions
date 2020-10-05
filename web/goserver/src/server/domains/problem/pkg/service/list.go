package service

import (
	"context"

	problemFind "server/db/pkg/handler/problem/find"

	kitendpoint "server/kit/endpoint"
)

type ListRequestData struct {
	Page int64 `json:"page"`
	Size int64 `json:"pageSize"`
}

func (s *basicProblemService) List(ctx context.Context, req ListRequestData, responseChan chan kitendpoint.Response) {
	problemFindRespChan := problemFind.Send(
		ctx,
		s.Conn,
		problemFind.RequestData{
			Page: req.Page,
			Size: req.Size,
		},
	)
	for r := range problemFindRespChan {
		responseChan <- r
		if r.IsLast {
			return
		}
	}
}

package endpoint

import (
	"context"

	"server/domains/build/pkg/service"
	kitendpoint "server/kit/endpoint"
)

type Endpoints struct {
	List             kitendpoint.Endpoint
	Create           kitendpoint.Endpoint
	CreateEmpty      kitendpoint.Endpoint
	UpdateAssetState kitendpoint.Endpoint
	UpdateTmps       kitendpoint.Endpoint
}

func New(s service.BuildService, mdw map[string][]kitendpoint.Middleware) Endpoints {
	eps := Endpoints{
		List:             MakeListEndpoint(s),
		Create:           MakeCreateEndpoint(s),
		CreateEmpty:      MakeCreateEmptyEndpoint(s),
		UpdateAssetState: MakeUpdateAssetStateEndpoint(s),
		UpdateTmps:       MakeUpdateTmpsEndpoint(s),
	}

	// for _, m := range mdw["WebSocket"] {
	// 	eps.Details = m(eps.Details)
	// }
	return eps
}

func MakeListEndpoint(s service.BuildService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.ListRequestData)
		return s.List(ctx, req)
	}
}

func MakeCreateEndpoint(s service.BuildService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			s.Create(ctx, req.(service.CreateRequestData))
			returnChan <- kitendpoint.Response{
				Data:   "OK",
				Err:    nil,
				IsLast: true,
			}
		}()
		return returnChan
	}
}

func MakeCreateEmptyEndpoint(s service.BuildService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			resp := s.CreateEmpty(ctx, req.(service.CreateEmptyRequestData))
			returnChan <- kitendpoint.Response{
				Data:   resp,
				Err:    nil,
				IsLast: true,
			}
		}()
		return returnChan
	}
}

func MakeUpdateAssetStateEndpoint(s service.BuildService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			resp := s.UpdateAssetState(ctx, req.(service.UpdateAssetStateRequestData))
			returnChan <- kitendpoint.Response{
				Data:   resp,
				Err:    nil,
				IsLast: true,
			}
		}()
		return returnChan
	}
}

func MakeUpdateTmpsEndpoint(s service.BuildService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			resp := s.UpdateTmps(ctx, req.(service.UpdateTmpsRequestData))
			returnChan <- kitendpoint.Response{
				Data:   resp,
				Err:    nil,
				IsLast: true,
			}
		}()
		return returnChan
	}
}

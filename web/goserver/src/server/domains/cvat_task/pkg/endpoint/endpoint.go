package endpoint

import (
	"context"

	"server/domains/cvat_task/pkg/service"
	kitendpoint "server/kit/endpoint"
)

type Endpoints struct {
	Dump         kitendpoint.Endpoint
	FindInFolder kitendpoint.Endpoint
	Setup        kitendpoint.Endpoint
}

func New(s service.CvatTaskService, mdw map[string][]kitendpoint.Middleware) Endpoints {
	eps := Endpoints{
		Dump:         MakeDumpEndpoint(s),
		FindInFolder: MakeFindInFolderEndpoint(s),
		Setup:        MakeSetupEndpoint(s),
	}

	// for _, m := range mdw["WebSocket"] {
	// 	eps.Details = m(eps.Details)
	// }
	return eps
}

func MakeDumpEndpoint(s service.CvatTaskService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		return s.Dump(ctx, req.(service.DumpRequestData))
	}
}

func MakeFindInFolderEndpoint(s service.CvatTaskService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		returnChan := make(chan kitendpoint.Response)
		go func() {
			defer close(returnChan)
			resp := s.FindInFolder(ctx, req.(service.FindInFolderRequestData))
			returnChan <- kitendpoint.Response{
				Data:   resp,
				Err:    kitendpoint.Error{Code: 0},
				IsLast: true,
			}
		}()
		return returnChan
	}
}

func MakeSetupEndpoint(s service.CvatTaskService) kitendpoint.Endpoint {
	return func(ctx context.Context, req interface{}) chan kitendpoint.Response {
		return s.Setup(ctx, req.(service.SetupRequestData))
	}
}
